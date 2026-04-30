---
title: "AlphaFold Series: distance map에서 geometric generator까지, 아키텍처는 어떻게 진화했나"
date: 2026-02-20 14:00:00 +0900
description: "AlphaFold 1, 2, 3의 진짜 차이는 성능 수치보다 계산 그래프에 있다. AF1은 distogram과 differentiable potential optimization, AF2는 Evoformer와 IPA 기반 end-to-end structure generation, AF3는 pairformer와 diffusion 기반 biomolecular coordinate generation으로 구조 예측의 중심 표현을 계속 바꿔 왔다."
categories: [AI, Protein Structure]
tags: [protein-structure, alphafold, series-summary, evoformer, diffusion, structure-prediction]
math: true
mermaid: false
image:
  path: /assets/img/posts/alphafold-series-summary/af-evolution.png
  alt: "Evolution of AlphaFold architectures from distogram prediction to diffusion-based universal structure generation"
---

## Hook

AlphaFold 시리즈를 성능 숫자만으로 보면 이야기가 너무 단순해진다. AF1은 잘했다, AF2는 혁명이었다, AF3는 범용화됐다. 물론 틀린 말은 아니다. 하지만 이 시리즈가 정말 흥미로운 이유는 각 세대가 **무엇을 더 잘 예측했는가**보다, **어떤 계산 그래프로 구조 문제를 다시 썼는가**에 있다.

AF1은 folding을 fragment assembly에서 learned geometric potential optimization으로 옮겼다. AF2는 그 learned geometry를 더 안쪽으로 밀어 넣어, Evoformer와 Invariant Point Attention을 통해 representation 자체가 구조를 접게 만들었다. AF3는 다시 그 구조 생성기를 단백질 전용 residue-frame 문법에서 떼어내, pairformer와 diffusion 기반 atom coordinate generator로 확장했다.

즉 이 시리즈의 진짜 궤적은 이런 식이다.

> **contact-like supervision → pairwise geometric landscape → end-to-end geometric reasoning → universal biomolecular coordinate generation**

이 글에서는 AlphaFold 1, 2, 3를 단순 요약이 아니라 **아키텍처 진화사**로 읽어보려 한다. 무엇이 버려졌고, 무엇이 유지됐고, 어떤 representation이 다음 세대로 넘어갔는지에 초점을 맞춘다.

## Problem

AlphaFold 시리즈 전체가 공통으로 겨냥한 질문은 결국 하나다.

> **서열과 관련 정보로부터 3D 생체분자 구조를 계산하려면, 어떤 representation과 generator가 가장 적절한가?**

세대를 따라가면 병목의 위치가 조금씩 바뀐다.

### 병목 1: hand-crafted search는 너무 비싸고 우회적이다

AF1 이전 주류는 fragment assembly였다. 구조를 직접 만드는 주 엔진이 stochastic search였고, learned signal은 많아야 보조 역할이었다. 이건 계산량이 크고, long-range consistency를 유지하기도 어렵다.

### 병목 2: pair geometry를 예측해도 structure realization이 별도 문제로 남는다

AF1은 learned distogram을 도입했지만, 여전히 structure realization은 별도 optimizer에 크게 의존했다. 즉 prediction과 generation 사이의 간극이 아직 존재했다.

### 병목 3: end-to-end 구조 생성이 되더라도 단백질 중심 표현에 갇힐 수 있다

AF2는 representation과 3D reasoning을 한 모델 안에 묶는 데 성공했지만, residue frame과 torsion-centric 설계는 본질적으로 protein-centric하다. 이건 ligand, nucleic acid, ion까지 다루는 데 한계가 된다.

### 병목 4: 범용화를 하려면 structure module 자체를 더 일반적인 생성기로 바꿔야 한다

AF3는 바로 이 문제를 건드린다. 범용 biomolecular complex prediction으로 가려면, structure head가 아니라 **atom coordinate generator 전체의 문법**이 바뀌어야 한다.

## Key Idea

세 세대의 아키텍처 변화를 압축하면 아래처럼 정리할 수 있다.

- **AlphaFold 1**
  - pairwise distance distribution을 예측한다.
  - 그 분포를 differentiable potential로 바꾼다.
  - gradient descent로 torsion space를 최적화한다.
- **AlphaFold 2**
  - MSA와 pair representation을 Evoformer에서 공동 업데이트한다.
  - IPA 기반 structure module이 residue frame 위에서 직접 3D 구조를 만든다.
  - recycling으로 learned iterative refinement를 수행한다.
- **AlphaFold 3**
  - trunk를 pair-centered pairformer로 이동시킨다.
  - structure module을 diffusion 기반 atom coordinate generator로 교체한다.
  - heterogeneous biomolecular complex를 하나의 생성 프레임워크로 다룬다.

핵심은 성능 향상 방식이 다르다는 점이다.

- AF1은 **better geometric target + learned optimization landscape**
- AF2는 **better representation coupling + internal 3D reasoning**
- AF3는 **better generator for heterogeneous coordinates**

## How It Works

### Architectural evolution at a glance

![AlphaFold evolution](/assets/img/posts/alphafold-series-summary/af-evolution.png)
_Figure 1: Conceptual evolution from distance prediction to universal biomolecular coordinate generation._

세 모델의 중심 계산 그래프를 아주 짧게 쓰면 이렇다.

```python
# AF1
features -> distogram -> differentiable potential -> L-BFGS on torsions -> structure

# AF2
sequence/MSA/templates -> Evoformer(msa, pair) -> IPA structure module -> structure

# AF3
sequence/MSA/templates/ligands -> Pairformer(pair, single) -> diffusion generator -> structure
```

겉으로는 다 비슷하게 “서열에서 구조를 예측”하는 것처럼 보이지만, 실제 중심축은 완전히 다르다.

### AlphaFold 1: learned potential architecture

AF1의 핵심 architecture는 2D residue-pair grid 위에 서 있다. 입력은 MSA, profile, covariation, Potts-like coupling feature다. 본체는 매우 깊은 residual convolutional network이고, 출력은 각 residue pair의 distance bin distribution이다.

이 모델의 architecture를 이해할 때 중요한 건, 이게 end-to-end coordinate generator가 아니라는 점이다. network의 역할은 structure 자체를 바로 생성하는 게 아니라 **protein-specific potential field를 예측하는 것**이다.

구조적으로 보면 AF1은 다음 세 부분으로 나뉜다.

1. feature extractor: MSA/covariation을 pair grid로 변환
2. deep 2D ResNet: distogram과 torsion-related signal 예측
3. differentiable optimizer: 예측된 potential을 실제 구조로 실현

즉 AF1의 진짜 generator는 network 단독이 아니라

- network + potential construction + optimizer

의 조합이다.

### AlphaFold 2: representation coupling architecture

AF2의 leap은 “더 좋은 distogram”이 아니다. 핵심은 MSA와 pair representation을 Evoformer 안에서 함께 업데이트한다는 점이다.

Evoformer를 architecture 관점에서 보면 다음과 같은 이중 스택이다.

- MSA stack
  - row attention with pair bias
  - column attention
  - transition
- pair stack
  - outer product mean으로 MSA 정보를 pair에 주입
  - triangle multiplicative update
  - triangle attention
  - pair transition

즉 AF2는 pairwise geometry를 독립 target으로만 다루지 않고, **representation 내부에서 sequence evidence와 geometry hypothesis를 교차 갱신**한다.

그 다음 Structure Module은 IPA를 통해 residue frame 위에서 3D reasoning을 수행한다. 이건 AF1과 본질적으로 다르다.

- AF1: geometry prediction 뒤 optimizer가 structure를 실현
- AF2: structure generation 자체가 모델 내부 module

AF2의 architecture identity는 한 줄로 정리하면 이거다.

> **geometry prediction과 coordinate generation을 하나의 learnable stack으로 연결한 첫 AlphaFold**

### AlphaFold 3: generator replacement architecture

AF3는 AF2를 그대로 넓힌 게 아니라, 오히려 구조 생성기의 문법 자체를 바꾼다. pairformer는 evoformer보다 MSA 의존성을 줄이고 pair relation 학습에 더 집중한다. 그리고 structure module 자리는 diffusion 기반 atom coordinate generator가 가져간다.

architecture 관점에서 가장 중요한 변화는 이렇다.

- AF2는 residue frame refinement module을 가진다.
- AF3는 atom coordinate distribution을 직접 생성하는 조건부 generator를 가진다.

이 차이는 매우 크다. 왜냐하면 ligand, nucleic acid, ion까지 다루려면 residue frame 기반 설계보다 atom-level coordinate generator가 훨씬 자연스럽기 때문이다.

AF3 diffusion module은 대략 이런 역할 분리를 가진다.

- atom encoder: noisy coordinates와 local chemistry 읽기
- token/pair conditioning: trunk context와 결합
- atom decoder: denoising update 생성
- iterative sampling loop: global and local geometry를 함께 복원

즉 AF3의 핵심은 trunk scaling이 아니라, **structure generator를 단백질 전용 refinement 모듈에서 범용 coordinate generator로 교체한 것**이다.

### Representation evolution

세 모델의 representation 중심도 다르다.

- **AF1:** pairwise geometry target이 중심
- **AF2:** MSA + pair 공동 representation이 중심
- **AF3:** pair + single + atom-level coordinate generation이 중심

더 직접적으로 쓰면,

- AF1은 `pair map predictor`
- AF2는 `representation-coupled structure generator`
- AF3는 `pair-conditioned universal coordinate generator`

라고 부르는 편이 실제 아키텍처 감각에 가깝다.

### What was kept, what was discarded

시리즈 전체에서 계속 유지된 것도 있다.

- pairwise relation이 핵심이라는 믿음
- structure prediction은 local signal만으론 안 된다는 점
- iterative refinement가 중요하다는 점

반대로 세대가 올라가며 버린 것도 분명하다.

- AF1에서 AF2로: 외부 optimizer 중심 generation
- AF2에서 AF3로: residue-frame 중심 protein-specific structure module
- 전반적으로: hand-crafted search와 overly explicit constraint engineering 비중 감소

즉 AlphaFold 시리즈는 “새로운 블록을 더한 역사”라기보다, **어떤 부분을 network 안으로 집어넣고 어떤 부분을 버렸는가의 역사**에 더 가깝다.

## Results

성능 수치도 중요하지만, 여기선 아키텍처 변화가 어떤 질적 성과로 이어졌는지를 보는 편이 더 낫다.

### AlphaFold 1

- fragment assembly 중심 질서를 실제로 흔들었다
- distogram prediction이 binary contact보다 훨씬 강한 supervision임을 보였다
- 하지만 structure realization은 아직 optimizer 의존성이 컸다

### AlphaFold 2

- CASP14에서 사실상 새로운 기준점을 세웠다
- learned representation coupling + internal structure generation이 실제로 먹힌다는 걸 증명했다
- structure prediction을 speculative modeling에서 usable geometry로 끌어올렸다

### AlphaFold 3

- protein-only를 넘어 heterogeneous biomolecular complex 전반으로 확장했다
- pair-centered trunk + diffusion generator 조합이 범용 complex modeling에 매우 강하다는 걸 보였다
- 구조 예측의 주 무대를 folding에서 interaction 쪽으로 더 이동시켰다

## Discussion

내가 보기엔 AlphaFold 시리즈 전체를 관통하는 가장 중요한 변화는 “생성기(generator)가 점점 모델 안쪽으로 들어오고, 점점 더 일반화된다”는 점이다.

- AF1에서는 generator의 상당 부분이 외부 optimizer였다.
- AF2에서는 구조 생성이 model-internal structure module로 들어왔다.
- AF3에서는 그 생성기가 다시 더 일반적인 diffusion coordinate generator로 바뀌었다.

이 흐름은 우연이 아니다. 구조 예측에서 truly hard한 부분이 단순 pairwise signal 예측이 아니라, **그 signal을 어떻게 일관된 3D object로 만들 것인가**였기 때문이다.

또 하나 흥미로운 건 representation의 중심 이동이다.

- AF1은 pairwise target을 잘 예측하는 게 핵심이다.
- AF2는 MSA와 pair가 대화하는 representation이 핵심이다.
- AF3는 pair relation을 중심으로 atom coordinate generation까지 포괄하는 더 큰 생성기가 핵심이다.

즉 AlphaFold 시리즈는 contact predictor가 발전한 역사라기보다, **geometric representation learning과 coordinate generation이 더 깊게 결합된 역사**라고 보는 편이 맞다.

## Limitations

### 1. 세대가 바뀔수록 범용성은 늘지만 해석 난이도도 올라간다

AF1은 pipeline이 길어도 직관적이다. AF2부터는 representation과 structure generation이 더 tightly coupled되고, AF3는 diffusion까지 들어가며 시스템 해석 난이도가 크게 올라간다.

### 2. 각 세대의 한계를 다음 세대가 모두 완전히 없애진 않는다

- AF1은 optimizer dependence가 컸다.
- AF2는 protein-centric bias가 강했다.
- AF3는 범용성이 커졌지만 diffusion sampling, confidence calibration, chemistry edge case라는 새 어려움이 있다.

즉 세대 교체는 “완전한 해결”보다 **병목의 위치 이동**으로 보는 게 더 정확하다.

### 3. 성능 비교만 보면 architecture lesson을 놓치기 쉽다

CASP score나 benchmark win만 보면 AF 시리즈를 leaderboard 역사로 읽게 된다. 하지만 진짜 배울 점은 각 세대가 **무엇을 intermediate로 두고, 무엇을 직접 생성 대상으로 삼았는가**에 있다.

## Conclusion

AlphaFold 시리즈의 핵심 진화는 성능 수치가 아니라 구조 예측 계산 그래프의 재배치에 있다. AF1은 learned distogram과 differentiable potential로 hand-crafted search를 약화시켰고, AF2는 Evoformer와 IPA로 representation과 coordinate generation을 하나의 모델 내부로 묶었으며, AF3는 pairformer와 diffusion으로 그 생성기를 더 범용적인 biomolecular coordinate engine으로 바꿨다.

내 기준에서 이 시리즈를 가장 잘 요약하는 문장은 이거다.

> **AlphaFold는 pairwise geometry를 더 잘 예측하게 된 것이 아니라, geometry를 실제 3D structure로 만드는 generator를 점점 더 내부화하고 일반화해 왔다.**

그래서 AF1, AF2, AF3는 단순 버전업이 아니다. 각각이 구조 예측 문제를 푸는 *아키텍처적 관점* 자체를 조금씩 바꾼 세 번의 전환점이다.

## TL;DR

- AF1의 중심은 **distogram + differentiable potential + gradient descent optimizer**다.
- AF2의 중심은 **Evoformer + IPA structure module + recycling**이다.
- AF3의 중심은 **pairformer + diffusion-based atom coordinate generator**다.
- 세 세대의 공통 축은 pair relation의 중요성이지만, structure generator의 위치와 형태는 계속 바뀐다.
- AF1은 learned geometry optimization, AF2는 end-to-end geometric reasoning, AF3는 universal biomolecular coordinate generation으로 읽는 편이 정확하다.
- 즉 AlphaFold 시리즈의 진짜 진화는 leaderboard보다 **generator architecture의 진화**에 있다.

## Paper Info

- **Scope:** AlphaFold 1, AlphaFold 2, AlphaFold 3 comparison
- **Focus:** architectural evolution, representation changes, and generator design
- **Related posts:**
  - AlphaFold 1 post
  - AlphaFold 2 post
  - AlphaFold 3 post

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다.
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
