---
title: "AlphaFold 3: biomolecular interaction 예측은 왜 structure module이 아니라 diffusion으로 갔나"
date: 2026-02-20 13:00:00 +0900
description: "AlphaFold 3는 단백질 전용 구조 예측기에서 벗어나 단백질, DNA, RNA, 리간드, 이온을 함께 다루는 biomolecular complex predictor로 확장됐다. 핵심은 AF2의 IPA 구조 모듈을 diffusion 기반 원자 좌표 생성기로 바꾸고, MSA 중심 trunk를 pair 중심으로 단순화한 데 있다."
categories: [AI, Protein Structure]
tags: [protein-structure, alphafold3, diffusion, biomolecular-interactions, drug-discovery, structure-prediction]
math: true
mermaid: false
image:
  path: /assets/img/posts/alphafold3-accurate-biomolecular-interactions/fig1.png
  alt: "AlphaFold 3 overview and performance across biomolecular interaction tasks"
---

## Hook

AlphaFold 2가 바꾼 건 단백질 구조 예측이었다. 그런데 실제 생물학과 약물개발은 단백질 하나로 끝나지 않는다. 우리가 진짜 알고 싶은 건 대개 **무엇이 무엇과 어떻게 붙는가**다. 단백질과 DNA, 단백질과 RNA, 항체와 항원, 단백질과 저분자 리간드, 심지어 금속 이온까지 포함한 복합체 구조가 핵심이다.

문제는 여기서부터였다. AF2는 단백질에는 거의 기념비적인 모델이었지만, 그 구조 모듈과 표현 방식은 꽤 강하게 단백질 중심이었다. residue frame, torsion angle, 단백질 서열 중심의 MSA 활용은 단백질에는 맞지만, arbitrary ligand chemistry나 heterogeneous complex 전체를 자연스럽게 포괄하기엔 한계가 있었다.

**AlphaFold 3**는 이 병목을 피하지 않고 정면으로 다시 설계한다. 핵심 질문은 단순하다.

> **단백질 전용 구조 예측기를 biomolecular interaction 전반의 범용 구조 생성기로 바꾸려면, 무엇을 버리고 무엇을 남겨야 하는가?**

이 논문이 내게 흥미로운 이유는 “AF2를 조금 키운 버전”이 아니라, 오히려 **AF2가 너무 단백질적이었던 부분을 걷어내고 더 일반적인 생성 모델 문법으로 다시 짠 시도**라는 점이다. 이 글에서는 AF3가 왜 diffusion으로 갔는지, 왜 evoformer보다 pairformer 쪽으로 무게를 옮겼는지, 그리고 이 변화가 drug discovery와 biomolecular modeling에 어떤 의미를 갖는지 풀어보겠다.

## Problem

AlphaFold 3가 푸는 문제는 “더 좋은 단백질 구조 예측”이 아니다. 더 정확히는 이거다.

> **여러 종류의 생체분자가 함께 있는 복합체를, 하나의 통합 프레임워크에서 높은 정확도로 예측할 수 있는가?**

이 문제는 적어도 네 가지 병목으로 나뉜다.

### 병목 1: AF2의 구조 모듈은 단백질에 너무 특화되어 있다

AF2의 structure module은 residue frame과 torsion angle이라는 매우 강한 단백질 inductive bias 위에 서 있다. 이건 단백질 backbone과 side-chain geometry를 다루는 데는 강력하지만, 리간드처럼 원자 조합이 훨씬 자유롭고 화학적 형태가 다양한 대상에는 그대로 확장하기 어렵다.

즉 문제는 단순히 입력 타입을 늘리는 게 아니다. **구조를 표현하는 좌표계 자체가 단백질 중심**이라는 점이 더 근본적이다.

### 병목 2: 실제 상호작용 문제는 heterogeneous entity를 동시에 다뤄야 한다

실전 biomolecular complex는 보통 이렇게 생긴다.

- protein only가 아니라
- protein + ligand,
- protein + DNA,
- protein + RNA,
- antibody + antigen,
- protein + ions,
- 또는 그 혼합

기존 도구들은 대개 영역별로 쪼개져 있었다.

- docking 도구는 protein-ligand에 집중하고
- nucleic-acid 복합체 모델은 또 따로 있고
- antibody나 multimer도 별도 특화 모델이 필요했다.

이런 분절형 접근은 workflow를 복잡하게 만들 뿐 아니라, **상호작용 유형 사이에 공유되는 구조적 규칙을 한 모델 안에서 재사용하기 어렵게** 만든다.

### 병목 3: MSA는 강력하지만, interaction 전반의 만능 정보원은 아니다

AF2의 강점 중 하나는 풍부한 진화정보를 evoformer에서 강하게 활용했다는 점이다. 그런데 biomolecular interaction 전체로 오면 이야기가 달라진다.

- 단백질 내부 진화 정보는 풍부할 수 있다.
- 하지만 단백질과 리간드 사이에는 MSA가 없다.
- 단백질과 DNA/RNA 사이의 cross-entity evolutionary signal도 제한적이다.

즉, interaction prediction 전체를 풀 때는 **MSA 중심 설계만으로는 일반화가 부족**하다. AF3가 pair representation 중심으로 무게를 옮기는 건 이 현실과 직접 연결된다.

### 병목 4: 물리적 타당성과 범용성을 동시에 잡아야 한다

리간드와 핵산까지 다루기 시작하면 local stereochemistry, bond geometry, clash avoidance 같은 문제는 오히려 더 까다로워진다. 단백질 전용 penalty나 hand-crafted constraint를 계속 덕지덕지 붙이면 확장은 더 어려워진다.

AF3의 핵심 고민은 결국 이거다.

> **단백질 특화 구조 제약을 줄이면서도, 화학적으로 말이 되는 복합체 좌표를 생성할 수 있는가?**

## Key Idea

AlphaFold 3의 핵심 아이디어는 세 줄로 요약할 수 있다.

1. **structure module을 diffusion 기반 원자 좌표 생성기로 교체한다.**
2. **MSA-heavy trunk 대신 pair 중심 representation learning으로 이동한다.**
3. **단백질, 핵산, 리간드, 이온을 하나의 atom-level generative framework 안에서 처리한다.**

기존 AF2와 대비하면 차이는 이렇게 정리된다.

- AF2
  - protein-centric structure module
  - residue frame + torsion angle representation
  - strong MSA/evoformer dependence
  - deterministic structure refinement에 가까운 흐름
- AF3
  - diffusion-based atom coordinate generation
  - heterogeneous biomolecular entities 지원
  - pair representation 중심 trunk
  - distribution에서 구조를 생성하는 generative framing

내가 보기엔 가장 중요한 변화는 “diffusion을 썼다”는 표면적 사실보다, **이걸 통해 구조 표현을 residue frame에서 atom coordinate generation으로 옮겼다**는 점이다. 이게 있어야 ligand, nucleic acid, ion까지 같은 언어로 다룰 수 있다.

## How It Works

### Overview

![AlphaFold 3 overview](/assets/img/posts/alphafold3-accurate-biomolecular-interactions/fig1.png)
_Figure 1: AlphaFold 3 overview and benchmark performance across heterogeneous biomolecular complexes. Source: original paper._

전체 흐름은 크게 세 단계다.

- 입력을 token / pair 수준으로 임베딩한다.
- pairformer trunk가 single/pair representation을 정제한다.
- diffusion module이 noisy atom coordinates를 점차 denoise하며 최종 구조를 만든다.

AF2와 비교하면 가장 큰 architectural message는 이렇다.

- MSA는 여전히 쓰지만 중심축이 아니다.
- pair representation이 trunk의 주인공이 된다.
- 최종 구조 생성은 residue frame refinement가 아니라 **원자 좌표 생성 문제**로 바뀐다.

아주 단순화한 pseudocode는 아래 정도로 볼 수 있다.

```python
# conceptual pseudocode
features = embed_inputs(sequences, msa, templates, ligands)
pair_repr, single_repr = run_pairformer(features)
coords = sample_gaussian_noise(n_atoms, 3)
for t in reversed(diffusion_steps):
    coords = denoise(coords, t, pair_repr, single_repr, features)
confidence = predict_confidence(pair_repr, coords)
return coords, confidence
```

핵심은 마지막 줄이 아니라 가운데 루프다. AF3는 구조를 한 번에 찍는 모델이라기보다, **noisy coordinate distribution에서 chemically plausible complex structure를 복원하는 생성기**에 가깝다.

### Representation: residue가 아니라 token과 atom으로 재정의

AF3는 입력을 residue-only 관점에서 보지 않는다. polymer residue와 ligand atom을 포함하는 보다 일반적인 tokenization을 쓴다. 단백질과 핵산은 residue 수준 token을 유지하지만, 리간드 쪽은 atom-level 정보가 훨씬 더 직접적으로 들어온다.

이건 매우 중요하다. 이유는 간단하다.

- 단백질은 residue-centric abstraction이 잘 먹힌다.
- 리간드는 residue라는 개념이 거의 무의미하다.
- 전체 복합체를 다루려면 결국 **heterogeneous entity를 같은 좌표 생성 문제 안에서 통합**해야 한다.

표현은 크게 두 축이다.

- **single representation**: 각 token의 로컬 특성
- **pair representation**: token-token 관계, 거리, 템플릿, 결합/상호작용 정보

이때 AF3가 더 강조하는 건 pair 쪽이다. interaction prediction의 본질이 “무엇이 무엇과 어떤 기하학으로 연결되는가”이기 때문이다.

### Pairformer: evoformer의 축소판이 아니라 관점 이동

![AlphaFold 3 architecture details](/assets/img/posts/alphafold3-accurate-biomolecular-interactions/fig2.png)
_Figure 2: Pairformer and diffusion-centered training setup. Source: original paper._

처음 보면 pairformer는 evoformer에서 MSA를 줄인 경량판처럼 보일 수 있다. 하지만 개념적으로는 조금 다르다. 이건 단순한 pruning이 아니라 **문제에 맞는 정보 우선순위의 재배치**다.

아키텍처 수준에서 보면 AF3 trunk는 대략 이렇게 읽을 수 있다.

- 입력 임베딩 단계에서 MSA, template, ligand/reference features를 합친다.
- 얕은 MSA 처리 후 중심 상태는 pair/single representation으로 넘어간다.
- 48개 수준의 pairformer block이 relation learning을 반복한다.
- structure 생성은 더 이상 trunk 안의 residue-frame refinement가 아니라 diffusion module에서 담당한다.

즉 AF3는 trunk와 generator의 역할을 더 선명하게 분리한다.

AF2에서는 진화 정보가 구조 예측의 핵심 동력 중 하나였다. 그러나 AF3가 겨냥하는 복합체 세계에서는 MSA가 항상 주연이 될 수 없다. 따라서 trunk는 더 보편적인 pair geometry 학습 쪽으로 무게를 옮긴다.

개념적으로 pairformer block은 이런 흐름으로 이해할 수 있다.

1. pair representation을 triangle-style relational updates로 정제한다.
2. pair 정보를 single 쪽으로 집계한다.
3. single representation을 self-attention으로 갱신하되 pair bias를 활용한다.
4. 갱신된 single 정보를 다시 pair 쪽으로 반영한다.

이 흐름의 직관은 명확하다. 복합체 예측에서 중요한 건 sequence-only 문법이 아니라 **entity 간 관계의 일관성**이다.

아주 단순화한 block sketch는 아래처럼 볼 수 있다.

```python
import torch
import torch.nn as nn

class PairformerBlock(nn.Module):
    def __init__(self, pair_dim: int, single_dim: int):
        super().__init__()
        self.pair_to_single = nn.Linear(pair_dim, single_dim)
        self.single_to_pair = nn.Linear(single_dim * 2, pair_dim)

    def forward(self, pair_repr, single_repr):
        pair_repr = pair_repr + triangle_update(pair_repr)
        single_repr = single_repr + self.pair_to_single(pair_repr.mean(dim=1))
        single_repr = single_repr + single_attention(single_repr, pair_bias=pair_repr)
        pair_repr = pair_repr + self.single_to_pair(pair_broadcast(single_repr))
        return pair_repr, single_repr
```

물론 실제 구현은 훨씬 복잡하지만, 구조적으로 보면 AF3는 **pair relation을 중심축으로 두고 single representation을 보조적으로 순환시키는 모델**이라고 보는 편이 맞다.

### Diffusion module: structure head가 아니라 generative coordinate engine

AF3에서 가장 눈에 띄는 변화는 diffusion module이다. 그런데 “요즘 다 diffusion 하니까” 수준으로 읽으면 핵심을 놓친다.

아키텍처 관점에서 diffusion module은 AF2의 structure module 자리를 단순 대체한 게 아니다. 오히려 역할이 더 커졌다.

- AF2 structure module은 residue frame refinement 모듈이다.
- AF3 diffusion module은 atom coordinate distribution 전체를 생성하는 조건부 generator다.

개념적으로 보면 내부 계산은 다음 흐름에 가깝다.

1. noisy atom coordinates를 받는다.
2. atom-level encoder가 local chemical context를 읽는다.
3. token-level transformer가 pair/single context와 결합한다.
4. atom decoder가 denoised coordinate update를 낸다.
5. 여러 timestep에 걸쳐 반복한다.

간단한 sketch는 아래처럼 쓸 수 있다.

```python
class DiffusionStructureGenerator(nn.Module):
    def forward(self, x_t, t, pair_repr, single_repr, atom_features):
        atom_state = atom_encoder(x_t, atom_features)
        token_state = fuse_token_context(atom_state, pair_repr, single_repr, t)
        delta = atom_decoder(token_state, atom_state)
        return denoise_update(x_t, delta, t)
```

즉 diffusion module은 AF2식 마지막 structure head보다 훨씬 더 **독립적인 generative subsystem**에 가깝다.

여기서 diffusion이 필요한 이유는 단지 generative trend 때문이 아니다. 더 중요한 이유는 다음 셋이다.

#### 1. atom coordinate를 직접 다루기 쉽다

리간드와 핵산까지 포함하면 residue frame 기반 parameterization은 금방 어색해진다. diffusion은 noisy coordinate를 denoise하는 방식이라 **원자 수준 좌표 생성**에 훨씬 자연스럽다.

#### 2. local geometry와 global packing을 함께 학습할 수 있다

낮은 noise regime에서는 local stereochemistry와 bond geometry를 잘 맞춰야 하고, 높은 noise regime에서는 전체 복합체 배치와 상호작용 배치를 복원해야 한다. diffusion은 이 다중 스케일 구조를 학습하기 좋은 프레임이다.

#### 3. hand-crafted penalty 의존도를 줄인다

AF2 식의 단백질 특화 구조 제약을 계속 늘리는 것보다, 모델이 데이터에서 일반적인 biomolecular geometry를 배우게 하는 쪽이 heterogeneous system에는 더 맞다.

정성적으로 쓰면 diffusion objective는 이런 감각이다.

$$
\mathcal{L}_{diff} = \mathbb{E}_{x,\epsilon,t}\big[\| \epsilon - \epsilon_\theta(x_t, t, c) \|^2\big]
$$

여기서 $x_t$ 는 noise가 섞인 atom coordinates, $c$ 는 pair/single/context representation이다. 논문 구현은 더 구조 특화돼 있지만, 큰 철학은 동일하다. **조건부 denoising을 통해 plausible complex structure manifold를 학습**하는 것이다.

### Confidence head와 ranking

구조를 하나 생성했다고 끝이 아니다. 실제 사용자는 어떤 샘플이 더 믿을 만한지도 알아야 한다. AF3는 confidence head를 통해 pLDDT, PAE 계열의 신호를 예측하고, 샘플 중 더 신뢰도 높은 구조를 고르는 데 활용한다.

이건 practical하게 중요하다. diffusion 기반 생성은 inherently sampling을 동반하므로, **좋은 구조를 만들기만 하는 게 아니라 그중 무엇이 더 맞을지 추정하는 체계**가 필요하다.

### Cross-distillation과 hallucination 억제

생성 모델은 구조가 잘 정의되지 않은 영역에서도 뭔가 그럴듯한 좌표를 만들어내고 싶어 한다. biomolecular modeling에서는 이게 오히려 문제다. 실제로는 floppy하거나 unstructured한 부분을 지나치게 정돈된 구조로 hallucinate할 수 있기 때문이다.

AF3는 이를 줄이기 위해 기존 AlphaFold-Multimer 예측을 distillation signal로 활용하는 전략을 쓴다. 논문의 포인트는 단순 성능 상승이 아니라, **불확실한 영역을 과하게 구조화하지 않도록 모델의 출력 습관을 교정**한다는 데 있다.

### 왜 이 설계가 먹히는가

AF3의 설계를 한 문장으로 요약하면 이렇다.

> **단백질 특화 구조 예측기를 범용 biomolecular coordinate generator로 바꾸기 위해, structure parameterization을 단순화하고 relation learning을 중심에 둔다.**

더 풀면,

- pairformer는 heterogeneous entity 간 관계를 trunk의 중심으로 둔다.
- diffusion은 atom-level geometry를 보다 일반적인 생성 문제로 바꾼다.
- confidence prediction은 샘플 기반 생성의 practical usability를 보완한다.

이 세 요소가 함께 있어야 AF3는 “AF2 + ligand head”가 아니라 **새로운 범용 interaction predictor**가 된다.

## Results

![AlphaFold 3 benchmark results](/assets/img/posts/alphafold3-accurate-biomolecular-interactions/fig3.png)
_Figure 3: Benchmark comparisons across protein-ligand, protein-nucleic-acid, and other biomolecular interaction tasks. Source: original paper._

결과를 볼 때는 “AF3가 정확하다”보다 **어떤 종류의 복합체에서 어떤 기준으로 강한가**를 봐야 한다.

### 1. heterogeneous interaction 전반에서 강하다

논문의 가장 큰 메시지는 개별 task SOTA 하나보다, **서로 다른 interaction class 전반에서 일관되게 높은 성능**을 낸다는 점이다.

- protein-ligand
- protein-DNA
- protein-RNA
- antibody-antigen
- 단백질 복합체 전반

즉, 여러 종류의 생체분자 상호작용을 각기 다른 툴 체인으로 풀던 흐름을 하나의 모델로 상당 부분 흡수할 가능성을 보여준다.

### 2. ligand docking 관점에서 특히 상징성이 크다

AF3가 protein-ligand 문제에서 강한 결과를 보인 건 단순 benchmark win 이상의 의미가 있다. structure biology와 drug discovery 실무에서는 결국 “리간드가 어디에 어떤 pose로 들어가는가”가 매우 중요하기 때문이다.

물론 AF3가 바로 medicinal chemistry workflow 전체를 대체한다는 뜻은 아니다. 하지만 최소한, **단백질 구조 예측 모델이 docking-quality interaction geometry까지 상당 수준 건드릴 수 있다**는 사실 자체가 큰 변화다.

### 3. 단일 프레임워크라는 점이 중요하다

내가 보기엔 AF3의 가장 강한 결과는 숫자 하나보다 **통합성**이다. 보통 이런 범용 모델은 범위는 넓지만 각 task에선 애매해지기 쉽다. AF3는 적어도 공개된 평가에서 “범용인데도 꽤 강하다”는 점을 설득력 있게 보여줬다.

### 4. 다만 benchmark 해석은 조심해야 한다

이런 대형 구조 예측 논문은 데이터 구성, leakage 방지, benchmark selection, confidence-based ranking 전략에 따라 체감 성능이 크게 달라질 수 있다. 그래서 “모든 docking을 끝냈다”처럼 읽는 건 과하다. 더 정확한 해석은 이쪽이다.

> **AF3는 biomolecular interaction prediction을 통합하는 데 매우 강한 baseline, 어쩌면 새로운 default를 제시했다. 하지만 특정 실무 domain에서의 전체 workflow superiority는 더 세밀한 검증이 필요하다.**

## Discussion

AF3는 단순히 AF2의 sequel이 아니라, 구조 생물학에서 deep learning 모델이 무엇을 출력해야 하는지에 대한 관점을 조금 바꿨다.

예전 관점은 이랬다.

- 단백질 서열이 핵심 입력이고
- 진화정보가 핵심 엔진이며
- 구조 모듈은 단백질 제약을 잘 반영해야 한다

AF3의 관점은 더 일반적이다.

- entity type은 더 다양해진다
- pair relation이 더 중요해진다
- 구조는 원자 좌표 생성 문제로 볼 수 있다
- 모델은 biomolecular geometry 전체를 generative하게 배운다

이건 꽤 중요한 이동이다. 왜냐하면 앞으로 구조 예측의 frontier는 단백질 folding 자체보다도 **interaction, binding, conditional complex formation, design loop integration** 쪽으로 더 이동할 가능성이 크기 때문이다.

특히 drug discovery 관점에서는 이 논문이 강하게 던지는 메시지가 있다. 이제 구조 예측 모델은 “apo structure를 잘 맞히는 도구”를 넘어서, **binding context를 이해하는 범용 model component**가 되기 시작했다는 점이다.

## Limitations

그렇다고 AF3를 만능 해결사로 보면 곤란하다. 한계는 분명하다.

### 1. static structure prediction과 full thermodynamics는 다르다

복합체 구조를 잘 맞힌다고 해서 binding free energy, kinetics, induced fit landscape, multiple metastable states를 다 해결하는 건 아니다. 구조 prediction은 중요한 축이지만, 물리 전체를 대체하진 않는다.

### 2. ligand chemistry의 practical edge cases는 여전히 어렵다

covalent binding, protonation ambiguity, tautomer issues, unusual coordination chemistry, solvent effects 같은 실전 chemistry 문제는 여전히 까다롭다. AF3의 공개 성능이 좋더라도 medicinal chemistry 현장에서는 더 세밀한 후처리나 별도 검증이 필요하다.

### 3. confidence가 곧 correctness는 아니다

confidence head는 매우 유용하지만, high confidence가 항상 biologically correct binding mode를 보장하진 않는다. 특히 데이터 분포 밖 사례나 유연성이 큰 시스템에서는 더 조심해야 한다.

### 4. 학습 데이터와 평가 세팅의 영향이 크다

이 정도 규모의 모델은 데이터 큐레이션과 split 전략의 영향을 많이 받는다. 어떤 종류의 generalization이 진짜로 강한지, 어떤 태스크에서 benchmark bias가 있는지는 계속 따져봐야 한다.

### 5. 접근성과 재현성 한계가 있다

AF 계열 모델은 영향력은 크지만, 완전한 재현과 대규모 실험은 일반 연구실 입장에선 여전히 부담이 크다. 실용적 파급력과 오픈 재현성 사이에는 간극이 남아 있다.

## Conclusion

AlphaFold 3의 핵심은 “단백질 구조 예측을 더 잘했다”가 아니다. 더 정확히는, **단백질 전용 구조 모듈에서 벗어나 biomolecular interaction 전체를 다루는 생성 모델로 재설계했다**는 데 있다. residue frame 기반 단백질 inductive bias를 줄이고, pair-centered trunk와 diffusion-based coordinate generation으로 옮긴 덕분에, 단백질, 핵산, 리간드, 이온이 섞인 복합체를 하나의 언어로 처리할 수 있게 됐다.

내 기준에서 이 논문의 진짜 의미는 구조 예측의 중심축이 folding에서 interaction으로 더 이동했다는 데 있다. AF2가 “단백질은 어떻게 접히는가”를 거의 바꿨다면, AF3는 “생체분자는 서로 어떻게 만나는가”를 더 일반적인 deep learning 문제로 바꿔놓는다.

물론 이것이 곧바로 모든 docking, allostery, binding energetics 문제를 끝냈다는 뜻은 아니다. 하지만 적어도 이제 biomolecular modeling의 기본 단위가 단백질 하나가 아니라 **heterogeneous complex 전체**라는 건 분명해졌다. 그런 의미에서 AF3는 단순한 버전업이 아니라, structural biology model design의 기준점을 한 번 더 옮긴 논문이다.

## TL;DR

- AlphaFold 3는 단백질 전용 구조 예측기에서 벗어나 **protein, DNA, RNA, ligand, ion을 함께 다루는 biomolecular complex predictor**로 확장됐다.
- 가장 큰 변화는 AF2의 IPA 기반 structure module을 버리고 **diffusion 기반 atom coordinate generator**로 바꾼 점이다.
- trunk도 MSA 중심 evoformer에서 **pair-centered pairformer** 쪽으로 무게를 옮겼다.
- 이 변화 덕분에 residue-frame 기반 단백질 특화 표현 대신, heterogeneous entity 전체를 더 일반적인 좌표 생성 문제로 다룰 수 있다.
- 공개 결과는 protein-ligand, nucleic-acid interaction, antibody-antigen 등 여러 interaction class 전반에서 매우 강한 성능을 보여준다.
- 다만 static structure accuracy가 thermodynamics 전체를 대체하는 것은 아니고, ligand chemistry edge case와 real-world generalization은 여전히 조심해서 봐야 한다.

## Paper Info

- **Title:** Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3
- **Authors:** Abramson et al.
- **Affiliations:** Google DeepMind and collaborators
- **Venue:** Nature
- **Published:** 2024-05-08
- **Paper:** https://www.nature.com/articles/s41586-024-07487-w
- **Project:** https://deepmind.google/discover/blog/alphafold-3-predicts-the-structure-and-interactions-of-all-of-lifes-molecules/

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다.
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
