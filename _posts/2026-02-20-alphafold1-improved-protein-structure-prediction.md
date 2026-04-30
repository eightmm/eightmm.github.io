---
title: "AlphaFold 1: contact prediction을 structure generation으로 바꾼 첫 번째 계산 그래프"
date: 2026-02-20 11:00:00 +0900
description: "AlphaFold 1은 binary contact를 넘어서 residue pair distance distribution을 예측하는 distogram과, 이를 단백질별 differentiable potential로 바꿔 gradient descent로 최적화하는 구조 생성 파이프라인을 제시했다. 핵심은 folding을 hand-crafted fragment sampling이 아니라 learned geometric potential optimization으로 재구성한 데 있다."
categories: [AI, Protein Structure]
tags: [protein-structure, alphafold1, distance-prediction, distogram, casp13, deep-learning]
math: true
mermaid: false
image:
  path: /assets/img/posts/alphafold1-improved-protein-structure-prediction/fig2.png
  alt: "AlphaFold 1 folding process and predicted structure refinement"
---

## Hook

AlphaFold 1을 지금 다시 보면, 사람들은 종종 이 모델을 “AF2 이전의 미완성 전단계” 정도로 기억한다. 그런데 그건 좀 아깝다. AF1은 단백질 구조 예측을 정말 중요한 방식으로 한 번 꺾어 놓은 논문이다. 그 전까지 free modeling의 기본 문법은 fragment assembly, hand-crafted statistical potential, 그리고 무거운 stochastic sampling 쪽에 더 가까웠다. AF1은 여기에 이렇게 묻는다.

> **굳이 fragment를 붙였다 떼었다 하며 구조를 찾지 말고, 네트워크가 단백질별 geometric potential을 직접 학습하게 하면 안 되나?**

이 질문은 생각보다 크다. 왜냐하면 구조 예측 문제를 “가능한 구조를 많이 샘플링해서 좋은 걸 고르는 문제”에서, **학습된 pairwise geometry를 연속 최적화로 실현하는 문제**로 바꾸기 때문이다.

AF2가 모든 스포트라이트를 가져간 건 당연하지만, AF1 없이는 AF2의 leap도 설명하기 어렵다. distogram, residue-pair representation, deep residual prediction, learned potential, gradient-based structure realization 같은 요소들은 전부 AF1에서 이미 강하게 등장한다. 이 글에서는 AF1이 기존 fragment assembly와 뭐가 달랐는지, 220-block ResNet이 정확히 어떤 역할을 했는지, 그리고 왜 이 모델이 “contact prediction 시스템”이 아니라 **structure generation 계산 그래프의 첫 버전**으로 읽혀야 하는지 정리해보겠다.

## Problem

AlphaFold 1이 겨냥한 문제는 단순히 “contact를 더 정확히 맞히자”가 아니다. 더 정확히는 다음과 같다.

> **MSA와 covariation으로 얻은 geometry signal을 실제 3D 구조로 효율적으로 실현하려면, 어떤 prediction target과 optimization pipeline이 필요한가?**

이 질문은 세 가지 병목으로 나뉜다.

### 병목 1: fragment assembly는 너무 비싸고 너무 우회적이다

AF1 이전 free modeling의 강자는 Rosetta 같은 fragment assembly 계열이었다. 이 방법은 대략 이렇게 작동한다.

- 짧은 fragment 라이브러리를 준비하고
- 구조를 stochastic move로 계속 바꾸고
- hand-crafted potential이 낮아지는 방향으로 샘플링한다

문제는 obvious하다.

- 계산량이 크다
- sampling quality가 결과를 많이 좌우한다
- long-range interaction을 만족하는 구조를 찾는 게 어렵다
- 좋은 geometry signal이 있어도 최적화가 그걸 잘 실현해준다는 보장이 약하다

즉 핵심 병목은 “좋은 구조가 없는 것”만이 아니라, **좋은 구조로 가는 검색 과정이 비효율적**이라는 점이다.

### 병목 2: binary contact는 구조적 정보가 너무 거칠다

contact prediction은 구조 예측을 크게 밀어 올렸지만, 본질적으로 coarse하다.

- 4 Å와 7.9 Å가 둘 다 contact다.
- 8.1 Å는 non-contact다.
- 하지만 구조적 의미는 이 셋이 연속적이지 binary하지도 않다.

즉 protein folding에 필요한 건 단순 yes/no edge가 아니라, **거리의 분포적 정보와 불확실성 자체**다. 이걸 더 풍부하게 예측해야 downstream structure realization이 좋아질 수 있다.

### 병목 3: geometry prediction과 structure realization이 분리되어 있다

많은 기존 접근은 contact나 distance-like constraint를 예측한 뒤, 그다음 별도 energy function이나 heuristic solver가 구조를 만든다. 이건 사실 두 문제다.

1. geometry를 잘 예측하는 문제
2. 그 geometry를 만족하는 3D structure를 찾는 문제

AF1의 핵심은 이 둘을 느슨하게 연결하지 않고, **예측된 분포를 곧바로 단백질별 differentiable potential로 바꿔 구조를 최적화**한다는 데 있다.

## Key Idea

AlphaFold 1의 핵심은 세 가지다.

1. **binary contact 대신 distogram을 예측한다.**
2. **예측 분포를 protein-specific differentiable potential로 바꾼다.**
3. **fragment assembly 대신 gradient descent로 torsion space를 최적화한다.**

핵심 흐름을 한 줄로 쓰면 이렇다.

> **MSA와 covariation을 입력으로 residue-pair distance distribution을 학습하고, 그 분포를 직접 structure optimization objective로 사용한다.**

이건 단순히 target을 바꾼 게 아니다. prediction과 realization 사이 연결 방식 전체가 달라진다.

- 입력: sequence + MSA + covariation features
- 중간 출력: residue pair distance distribution
- 구조 생성: learned potential 위에서 torsion angle optimization

AF1은 여전히 fully end-to-end model은 아니지만, 구조 예측 파이프라인의 중심을 **hand-crafted search**에서 **learned geometry + differentiable optimization**으로 옮긴 첫 번째 강한 사례다.

## How It Works

### Overview

![AlphaFold 1 overview](/assets/img/posts/alphafold1-improved-protein-structure-prediction/fig2.png)
_Figure 1: AlphaFold 1 folding process and refinement behavior. Source: original paper._

AF1 파이프라인은 크게 네 단계다.

- MSA와 covariation feature를 만든다.
- deep 2D residual network가 distogram과 torsion distribution을 예측한다.
- distogram을 differentiable potential로 바꾼다.
- backbone torsion angle 공간에서 L-BFGS 기반 최적화를 반복해 구조를 실현한다.

아주 압축한 pseudocode는 아래 정도다.

```python
# conceptual pseudocode
msa = build_msa(sequence)
features = extract_pair_and_sequence_features(sequence, msa)
distogram, torsion_dist = predict_geometry(features)
potential = build_structure_potential(distogram, torsion_dist)
for restart in range(num_restarts):
    phi, psi = initialize_torsions(torsion_dist)
    phi, psi = optimize_with_lbfgs(potential, phi, psi)
return best_structure_over_restarts()
```

여기서 중요한 건 AF1이 structure를 네트워크가 바로 내놓지 않는다는 점이다. 대신 네트워크는 **구조를 향해 미분 가능한 지형(landscape)** 을 만들고, 최적화기가 그 지형 위를 내려간다.

### Input representation: MSA를 feature bank로 쓴다

AF1의 입력은 단순 서열 one-hot이 아니다. 이 모델은 MSA와 covariation을 아주 적극적으로 feature화한다.

대표적으로 들어가는 신호는 다음과 같다.

- amino acid identity / sequence profile
- HHblits / PSI-BLAST 기반 profile
- deletion / gap 관련 정보
- Potts model 계수 같은 covariation signal
- residue pair 수준의 통계적 coupling

즉 AF1은 Transformer-style raw sequence model이라기보다, **당시 구조예측 커뮤니티가 중요하게 보던 진화/통계 feature를 고밀도로 집어넣은 supervised geometric predictor**다.

이 표현의 포인트는 분명하다.

- local sequence identity만으로는 부족하다
- MSA는 contact와 fold에 관한 long-range signal을 담고 있다
- pairwise coupling을 network가 직접 이용할 수 있어야 한다

### Distogram network: AF1의 핵심 아키텍처

![AlphaFold 1 architecture](/assets/img/posts/alphafold1-improved-protein-structure-prediction/fig3.png)
_Figure 2: AlphaFold 1 network and distogram prediction setup. Source: original paper._

AF1의 중심은 residue-pair grid를 입력으로 받는 **매우 깊은 2D residual convolutional network**다. 핵심은 “서열을 임베딩해서 1D로 처리한 뒤 pair를 붙이는” 식이 아니라, **처음부터 구조 문제를 2D pair map 문제로 본다**는 점이다.

논문에서 중요한 특징은 다음과 같다.

- 입력은 residue × residue pair map
- 깊은 residual stack 사용
- dilation을 활용해 receptive field를 넓힘
- 출력은 각 residue pair에 대한 distance bin 분포

즉 구조적으로 보면 AF1의 본체는 “pairwise image segmentation network”에 가깝다. 단백질 전체를 그래프나 3D object로 직접 다루기보다, **모든 residue pair를 채널이 많은 2D tensor 위에서 계산**한다.

개념적 스케치는 이렇다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistogramResNet(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, bins: int = 64):
        super().__init__()
        self.input_proj = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualPairBlock(hidden, dilation=d)
            for d in [1, 2, 4, 8] * 12
        ])
        self.out = nn.Conv2d(hidden, bins, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        logits = self.out(x)
        return logits.softmax(dim=1)

class ResidualPairBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = F.elu(self.conv1(x))
        h = F.elu(self.conv2(h))
        h = self.conv3(h)
        return x + h
```

실제 AF1은 훨씬 더 큰 스택과 세부 설계를 쓰지만, 중요한 건 계산 그래프의 철학이다. **구조 문제를 residue-pair interaction field 위에서 푼다**는 관점이 이미 여기서 매우 선명하다.

### Distogram이 왜 중요했나

AF1의 distogram은 residue pair $(i, j)$마다 하나의 숫자를 내는 게 아니라, 거리 bin 전체에 대한 확률분포를 낸다.

$$
P(d_{ij} \mid S, \mathrm{MSA}(S))
$$

이 분포형 출력은 세 가지 이점이 있다.

1. binary contact보다 훨씬 풍부하다.
2. uncertainty를 함께 표현할 수 있다.
3. downstream potential로 바꿀 때 더 부드럽고 미분 가능한 구조를 만들기 쉽다.

결국 AF1은 contact map을 맞히는 모델이 아니라, **protein-specific geometric landscape를 예측하는 모델**이다.

### Potential construction: 예측을 에너지 지형으로 바꾼다

AF1의 진짜 아이디어는 여기서 완성된다. distogram은 최종 출력이 아니라, 구조 최적화를 위한 potential로 변환된다.

개념적으로는 residue pair distance가 특정 값일수록 확률이 높다면, 그 구조는 더 좋은 구조여야 한다. 따라서 negative log probability를 potential처럼 사용할 수 있다.

$$
V_{dist}(\phi, \psi) = \sum_{i,j} -\log P\big(d_{ij}(\phi, \psi)\big)
$$

여기서 $d_{ij}(\phi, \psi)$ 는 torsion angle이 결정하는 실제 residue pair distance다. 여기에 torsion prior와 van der Waals 같은 보조 항을 더해 전체 목적함수를 만든다.

$$
V_{total}(\phi, \psi) = V_{dist} + V_{torsion} + V_{vdW}
$$

이 단계가 AF1의 architectural identity를 결정한다. network가 structure를 직접 출력하진 않지만, **structure가 따라야 할 objective function 자체를 prediction으로 만든다**는 점에서 굉장히 현대적이다.

### Structure realization: fragment assembly 대신 gradient descent

AF1은 backbone torsion angle $(\phi, \psi)$ 를 최적화 변수로 두고, L-BFGS 같은 gradient-based optimizer로 potential을 최소화한다. 즉 structure generation은 다음처럼 읽을 수 있다.

- 초기 torsion angle을 샘플링한다.
- 예측된 potential 위에서 연속 최적화를 한다.
- 여러 restart를 돌려 낮은 potential basin을 찾는다.
- 가장 좋은 구조를 고른다.

여기서 중요한 건 AF1이 sampling을 완전히 없애진 않았다는 점이다. multi-start optimization이 여전히 필요하다. 하지만 sampling의 역할이 바뀐다.

- 예전: 구조를 만드는 주된 엔진이 stochastic search
- AF1: 구조는 learned potential이 만들고, sampling은 local minimum 탐색 보조 수단

이건 큰 차이다.

### Torsion head와 local geometry

AF1은 distogram만 예측하는 게 아니라 torsion 관련 분포도 함께 다룬다. 이건 global fold만 맞는 구조를 피하고, local backbone geometry를 더 안정적으로 잡는 데 도움이 된다. 다시 말해 AF1은 처음부터 “장거리 pair geometry”와 “국소 backbone prior”를 함께 쓰는 혼합 설계다.

이 역시 이후 AF2의 구조 모듈로 가는 징검다리처럼 읽힌다. 완전히 end-to-end는 아니지만, **구조적 prior와 geometry signal을 하나의 계산 흐름으로 묶으려는 방향**은 이미 뚜렷하다.

### 왜 이 설계가 먹혔나

AF1의 설계를 한 문장으로 요약하면 이렇다.

> **deep pairwise geometry predictor가 만든 분포를 단백질별 에너지 지형으로 바꾸고, 그 지형 위에서 연속 최적화를 수행한다.**

더 풀면,

- MSA/covariation은 pairwise geometric evidence를 준다.
- deep 2D ResNet은 그 evidence를 distogram으로 정리한다.
- distogram은 differentiable potential로 바뀐다.
- optimizer는 그 potential을 만족하는 structure를 torsion space에서 찾는다.

이 구조 덕분에 AF1은 단순 contact classifier에서 벗어나, **예측과 구조 생성이 직접 연결된 pipeline**이 된다.

## Results

![AlphaFold 1 results](/assets/img/posts/alphafold1-improved-protein-structure-prediction/fig4.png)
_Figure 3: AlphaFold 1 benchmark behavior and CASP13-level performance. Source: original paper._

### 1. CASP13에서 질적으로 다른 점프를 보였다

AF1은 CASP13 free modeling에서 매우 강한 결과를 냈다. 중요한 건 absolute score 하나보다, **deep learning 기반 geometry prediction이 fragment assembly 중심 질서를 실제로 흔들었다**는 점이다.

### 2. distogram은 contact보다 훨씬 유용했다

결과적으로 AF1은 binary contact를 넘어 분포형 distance target이 얼마나 강한 supervision인지 보여줬다. 이후 구조 예측 모델들이 pairwise distance / geometry를 훨씬 더 본격적으로 다루게 된 건 우연이 아니다.

### 3. 하지만 최종 structure generation은 아직 optimizer 의존적이었다

AF1은 분명 강력했지만, structure realization이 여전히 별도 optimization 단계에 많이 기대고 있었다. 이건 장점이자 한계였다.

- 장점: 예측된 geometry를 꽤 잘 실현했다
- 한계: pipeline이 완전히 end-to-end는 아니었다

바로 이 지점이 AF2가 다음에 해결하려는 병목이 된다.

## Discussion

내가 보기에 AF1의 가장 중요한 유산은 “deep learning이 structure prediction을 잘했다”가 아니다. 더 중요한 건, **구조 예측 파이프라인의 중심 물음을 바꿨다**는 점이다.

예전엔 보통 이렇게 생각했다.

- 좋은 statistical potential을 설계하고
- 충분히 많이 샘플링하면
- 구조가 나온다

AF1은 반대로 말한다.

- potential 자체를 단백질별로 학습하게 하자
- geometry signal을 더 풍부한 분포 형태로 예측하자
- optimizer는 그 학습된 지형을 따라가게 하자

이 관점 전환은 매우 중요했다. 왜냐하면 AF2의 end-to-end 구조 예측도 결국 이 질문 위에서 자라났기 때문이다. AF2는 AF1의 learned potential + optimization 구도를 더 안쪽으로 밀어 넣어, 아예 모델이 representation 안에서 structure를 만들게 한다.

즉 AF1은 구식 AlphaFold가 아니라, **contact prediction 시대와 end-to-end structure generation 시대 사이를 잇는 결정적 브리지**다.

## Limitations

### 1. structure realization이 여전히 외부 최적화에 의존한다

AF1의 가장 큰 한계는 여전히 명확하다. 네트워크는 structure를 직접 생성하지 않고, optimizer가 많이 책임진다. 그래서 pipeline이 길고, restart와 local minima 문제도 남는다.

### 2. 단백질 전용 설계다

표현과 목적함수 모두 단백질 backbone torsion 기반이다. ligand, nucleic acid, heterogeneous complex 같은 문제로 자연스럽게 확장되긴 어렵다.

### 3. global consistency는 아직 완전히 내부화되지 않았다

distogram이 풍부해도, 그걸 3D structure로 일관되게 실현하는 건 여전히 쉽지 않다. AF2가 triangle update, IPA, recycling 같은 구조를 도입한 이유가 바로 여기 있다.

## Conclusion

AlphaFold 1은 AF2의 거대한 그림자에 가려져 있지만, 실제로는 구조 예측 계산 그래프를 처음으로 크게 비튼 논문이다. binary contact를 residue-pair distance distribution으로 바꾸고, 그 분포를 단백질별 differentiable potential로 변환한 뒤, fragment assembly 대신 gradient descent로 구조를 최적화한다. 이 조합은 folding을 hand-crafted search problem에서 **learned geometry optimization problem**으로 바꿔 놓는다.

내 기준에서 AF1의 진짜 의미는 성능 그 자체보다, 구조 예측 문제의 내부 표현을 바꾼 데 있다. 이후 AF2는 이 흐름을 더 밀어붙여 representation과 좌표 생성을 end-to-end로 묶고, AF3는 그걸 다시 biomolecular complex 전반으로 확장한다. 그런 의미에서 AF1은 “전 단계 모델”이 아니라, **AlphaFold 시리즈 전체의 첫 번째 아키텍처적 전환점**이다.

## TL;DR

- AlphaFold 1은 binary contact 대신 **distogram**을 예측했다.
- 핵심 network는 residue-pair grid 위에서 작동하는 **깊은 2D residual convolutional architecture**다.
- 예측된 distance distribution은 **protein-specific differentiable potential**로 바뀐다.
- structure generation은 fragment assembly 대신 **torsion angle space에서의 gradient descent optimization**으로 수행된다.
- 즉 AF1의 핵심은 contact prediction 자체보다, **learned geometry를 직접 optimization landscape로 바꾼 것**이다.
- 다만 structure realization이 여전히 외부 optimizer에 크게 의존해, 완전한 end-to-end 구조 생성은 아니었다.
- 바로 그 한계가 AF2의 Evoformer + Structure Module 설계로 이어진다.

## Paper Info

- **Title:** Improved Protein Structure Prediction Using Potentials from Deep Learning
- **Authors:** Senior et al.
- **Affiliations:** DeepMind and collaborators
- **Venue:** Nature
- **Published:** 2020-01-15
- **Paper:** https://www.nature.com/articles/s41586-019-1923-7

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다.
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
