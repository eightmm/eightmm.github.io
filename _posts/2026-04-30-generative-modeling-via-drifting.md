---
title: "Generative Modeling via Drifting: one-step 생성은 inference가 아니라 training에서 만들어진다"
date: 2026-04-30 11:40:00 +0900
description: "Drifting Models는 diffusion/flow의 iterative inference를 training-time pushforward evolution으로 옮긴다. anti-symmetric drifting field와 drift regression objective로 one-step generation을 학습하고, ImageNet 256x256에서 1-NFE FID 1.54(latent), 1.61(pixel)을 기록한다."
categories: [AI, Generative Models]
tags: [generative-modeling, diffusion, flow-matching, one-step-generation, drifting-model, pushforward, imagenet, fid]
math: true
mermaid: false
image:
  path: /assets/img/posts/generative-modeling-via-drifting/fig1_overview.png
  alt: "Drifting Models overview and generated ImageNet samples"
---

## Hook

diffusion이나 flow matching 논문을 읽다 보면 점점 비슷한 질문으로 돌아오게 된다. **정말 inference 때 여러 step을 돌아야만 좋은 샘플이 나오나?** 지금까지의 주류 답은 대체로 yes였다. 조금 더 정확히 말하면, 복잡한 분포 매칭을 한 번에 하기 어려우니 noise에서 data로 가는 경로를 잘게 쪼개서, inference에서 여러 번 네트워크를 호출하는 쪽이 더 안정적이라는 믿음이었다.

이 논문 **Generative Modeling via Drifting**은 그 기본 전제를 뒤집는다. 저자들의 문제 설정은 꽤 직설적이다. 생성 모델의 본질은 결국 prior 분포를 data 분포로 보내는 pushforward map을 배우는 일인데, 굳이 그 분포 진화를 *inference time* 에서 수행할 필요가 있느냐는 것이다. 어차피 딥러닝 학습 자체가 반복적 최적화 과정이라면, 분포의 진화는 **training time** 에서 일어나게 하고, inference는 그냥 한 번의 forward pass로 끝내자는 발상이다.

이게 단순한 철학적 말장난으로 끝났다면 재미없는 아이디어였을 텐데, 논문은 여기에 꽤 구체적인 수학적 장치를 붙인다. generated distribution과 data distribution 사이의 차이를 sample movement로 바꾸는 **drifting field** 를 정의하고, 이 drift가 0이 되는 equilibrium을 학습 목표로 삼는다. 결과적으로 모델은 one-step generator인데, 학습 중에는 분포 자체가 계속 움직이며 target distribution 쪽으로 수렴한다.

흥미로운 건 성능도 그냥 “개념 증명” 수준이 아니라는 점이다. 논문은 ImageNet 256x256에서 **1-NFE FID 1.54(latent)**, **1-NFE FID 1.61(pixel)** 를 보고한다. 즉, “한 번에 뽑는 생성은 품질이 떨어진다”는 오래된 직관을 정면으로 압박한다. 이 글에서는 drifting이 정확히 뭘 뜻하는지, 왜 anti-symmetry가 핵심인지, 그리고 이 접근을 어디까지 믿어도 되는지를 차분히 정리해보겠다.

## Problem

이 논문이 겨냥하는 문제는 단순히 “one-step generation을 더 잘하자”가 아니다. 좀 더 본질적인 질문은 이거다.

> **분포 매칭을 위해 필요한 iterative dynamics를 inference가 아니라 training으로 옮길 수 있는가?**

이 질문이 중요한 이유는 기존 생성 모델의 계산 구조 때문이다.

### 병목 1: diffusion/flow의 품질은 좋지만 inference cost가 크다

현대 생성 모델의 가장 강한 축은 diffusion과 flow 계열이다. 둘 다 핵심은 비슷하다.

- 현재 샘플을 조금 더 data-like 하게 바꾸는 작은 업데이트를 반복한다.
- 이 작은 업데이트는 신경망이 계산한다.
- 결과적으로 샘플 품질은 좋지만, inference에서 네트워크를 여러 번 불러야 한다.

즉, 좋은 분포 매칭은 가능하지만, 계산은 비싸다. 최근엔 distillation이나 consistency 계열처럼 step 수를 줄이려는 시도가 많았지만, 상당수는 여전히 multi-step teacher나 trajectory approximation에 기대고 있다.

### 병목 2: one-step generation은 쉽게 mode collapse나 weak supervision으로 흐른다

그렇다면 그냥 GAN처럼 한 번에 뽑으면 되지 않나 싶지만, one-step generator는 다른 어려움이 있다.

- adversarial training은 불안정할 수 있다.
- 직접적인 density/path supervision이 없으면 mode coverage가 흔들리기 쉽다.
- 생성 분포 전체를 어떻게 움직여야 하는지에 대한 구조적 힌트가 약하다.

논문은 바로 이 지점을 건드린다. 샘플 하나하나를 독립적으로 맞추는 게 아니라, **현재 생성 분포가 data 분포에 비해 어디로 drift해야 하는지**를 계산하면, one-step generator도 distribution-level signal을 받을 수 있다는 것이다.

### 병목 3: 생성은 결국 “함수 대 함수” 문제다

분류는 대개 샘플을 label로 보내는 문제다. 하지만 생성은 prior distribution을 data distribution으로 보내는 문제다. 즉,

$$
q = f_{\#} p_{\epsilon}
$$

에서 $q \approx p_{data}$ 가 되도록 하는 함수를 학습해야 한다. 이건 sample-to-label 문제보다 훨씬 어렵다. 왜냐하면 우리가 맞춰야 하는 대상이 개별 점이 아니라 **분포 전체의 구조**이기 때문이다.

Drifting Models는 이걸 “분포의 차이를 sample-level drift로 환원”하는 방식으로 다룬다.

## Key Idea

핵심 아이디어는 한 문장으로 요약된다.

> **생성 분포를 inference에서 조금씩 진화시키지 말고, training 중 모델 업데이트를 통해 진화시키자.**

이를 위해 논문은 세 가지를 묶는다.

1. **Pushforward at training time**
   - 모델 $f_\theta$ 가 바뀌면, 같은 noise $\epsilon$ 에 대한 출력 $x = f_\theta(\epsilon)$ 도 바뀐다.
   - 즉 SGD step 자체가 샘플을 drift시키는 작용으로 볼 수 있다.

2. **Drifting field $V_{p,q}(x)$**
   - 현재 generated distribution $q$ 와 target distribution $p$ 를 바탕으로, 샘플 $x$ 가 어느 방향으로 움직여야 하는지 벡터를 준다.
   - 이상적으로는 $q=p$ 일 때 drift가 0이어야 한다.

3. **Drift regression objective**
   - 현재 샘플 $x$ 를 drifted target $x + V(x)$ 쪽으로 회귀시키는 식으로 학습한다.
   - 이렇게 하면 optimizer가 분포를 점차 target 쪽으로 이동시킨다.

기존 패러다임과 비교하면 차이는 아래처럼 정리된다.

- Diffusion / Flow
  - 분포 진화 시점: inference time
  - inference 계산: 다단계 네트워크 호출
  - 학습 신호: score, velocity, path supervision
  - 핵심 구조: ODE/SDE dynamics
- Drifting Models
  - 분포 진화 시점: **training time**
  - inference 계산: **single forward pass**
  - 학습 신호: **drifting field based fixed-point regression**
  - 핵심 구조: **equilibrium-seeking drift dynamics**

좋은 점은 메시지가 꽤 선명하다는 것이다. 논문은 “더 빠른 diffusion”을 제안하는 게 아니라, **iterative generation 자체를 training 쪽으로 재배치**하는 완전히 다른 framing을 제안한다.

## How It Works

### Overview

![Drifting Models overview](/assets/img/posts/generative-modeling-via-drifting/fig1_overview.png)
_Figure 1: Drifting Models overview. The model learns a one-step generator, while the pushforward distribution evolves during training toward the data distribution. Source: project page / paper authors._

전체 과정을 아주 단순하게 요약하면 이렇다.

- noise를 한 번 generator에 넣어 sample을 만든다.
- real sample과 generated sample을 함께 써서 drifting field를 계산한다.
- 현재 sample을 drifted target 쪽으로 당기는 regression loss를 건다.
- optimizer update가 반복되면서 pushforward distribution 전체가 data 쪽으로 이동한다.

겉모습만 보면 BYOL류 fixed-point regression이나 self-distillation의 감각도 조금 있다. 하지만 목적은 representation learning이 아니라 **generated distribution의 movement** 다.

### Pushforward at training time

noise $\epsilon \sim p_\epsilon$ 를 generator에 넣으면 output $x=f(\epsilon)$ 가 나온다. 이 출력의 분포를 $q$ 라고 두면,

$$
q = f_{\#} p_\epsilon
$$

이다. 여기서 중요한 건 학습 iteration $i$ 마다 모델이 $f_i$ 로 바뀐다는 점이다. 그러면 같은 noise에 대해서도

$$
x_i = f_i(\epsilon), \qquad x_{i+1} = f_{i+1}(\epsilon)
$$

가 되고, 그 차이

$$
\Delta x_i = f_{i+1}(\epsilon) - f_i(\epsilon)
$$

를 training-induced drift로 볼 수 있다. 즉, 논문은 SGD update를 단순 parameter optimization이 아니라 **sample distribution evolution** 으로 해석한다.

이 시점에서 관점 전환이 일어난다. diffusion은 inference 단계에서 $x_t$ 를 업데이트하지만, drifting은 training 단계에서 $f_i$ 가 바뀌면서 $x_i$ 가 업데이트된다.

### Drifting field와 equilibrium

이제 필요한 건 “어디로 움직여야 하는가”를 알려주는 규칙이다. 논문은 이를 drifting field $V_{p,q}(x)$ 로 둔다.

$$
x_{i+1} = x_i + V_{p,q_i}(x_i)
$$

이 field는 data distribution $p$ 와 current generated distribution $q$ 둘 다에 의존한다. 가장 중요한 요구사항은 다음이다.

$$
q=p \Rightarrow V_{p,q}(x)=0
$$

즉, 생성 분포가 이미 data와 일치한다면 더 이상 drift할 필요가 없어야 한다. 논문은 이를 보장하기 위한 충분한 구조로 **anti-symmetry** 를 둔다.

$$
V_{p,q}(x) = -V_{q,p}(x)
$$

이 조건이 있으면 $p=q$ 일 때 자동으로 $V=0$ 이 된다. 직관적으로는 “positive와 negative의 역할을 뒤집으면 drift 방향이 반대로 바뀌어야 한다”는 뜻이다. 이 논문에서 anti-symmetry는 그냥 예쁜 수학 조건이 아니라, **equilibrium이 제대로 정의되도록 만드는 구조적 핵심** 이다.

### Attraction minus repulsion

논문이 실제로 쓰는 drifting field는 mean-shift 류의 attraction/repulsion 해석을 가진다. 샘플 $x$ 에 대해,

- real data 쪽 positive sample은 $x$ 를 끌어당기고,
- generated sample 쪽 negative sample은 $x$ 를 밀어낸다.

이를 간단히 쓰면,

$$
V_{p,q}(x) = V_p^+(x) - V_q^-(x)
$$

이다. 그리고 각 항은 kernel-weighted mean shift 비슷한 꼴이다.

$$
V_p^+(x) = \frac{1}{Z_p} \mathbb{E}_{y^+ \sim p}[k(x,y^+)(y^+ - x)]
$$

$$
V_q^-(x) = \frac{1}{Z_q} \mathbb{E}_{y^- \sim q}[k(x,y^-)(y^- - x)]
$$

직관은 꽤 납득 가능하다.

- data 쪽에서 비슷한 샘플들이 있으면 그 방향으로 당겨진다.
- generated 쪽에서 이미 자기 주변을 차지한 샘플들이 있으면 그쪽으론 덜 가게 만든다.

그래서 mode collapse를 막는 감각도 자연스럽게 나온다. generated samples가 한 mode에 뭉쳐 있으면, 다른 data mode가 attraction을 제공해 계속 끌어당길 수 있기 때문이다.

논문이 쓰는 kernel은 거리 기반 exponential 형태다.

$$
k(x,y) = \exp\left(-\frac{1}{\tau}\|x-y\|\right)
$$

실제 구현은 softmax normalization으로 들어간다. InfoNCE 느낌의 local similarity weighting으로 보면 된다.

### Training objective

학습은 surprisingly simple하다. 현재 출력 $x=f_\theta(\epsilon)$ 에서 drifted target

$$
x_{drifted} = \text{stopgrad}(x + V(x))
$$

를 만들고, 원래 출력이 이 target을 따라가도록 회귀한다.

$$
\mathcal{L} = \mathbb{E}_{\epsilon}\left[ \| f_\theta(\epsilon) - \text{stopgrad}(f_\theta(\epsilon)+V(f_\theta(\epsilon))) \|^2 \right]
$$

이 식이 좋은 이유는 두 가지다.

1. drift 자체의 norm을 줄이는 fixed-point objective가 된다.
2. distribution-dependent quantity인 $V$ 에 직접 미분을 깊게 걸지 않고, frozen target regression으로 안정화한다.

개념적인 pseudocode는 아래 정도다.

```python
# conceptual pseudocode
noise = randn(batch, c)
x = generator(noise)
y_neg = x
V = compute_drifting_field(x, y_pos=real_batch, y_neg=y_neg)
x_target = stopgrad(x + V)
loss = mse(x, x_target)
loss.backward()
optimizer.step()
```

이 구조는 표면적으로 매우 단순하지만, 실제 의미는 “generator output을 직접 정답에 맞춘다”가 아니라 **현재 분포가 가야 할 방향으로 한 step 옮긴 frozen target을 만든다**는 데 있다.

### Feature-space drifting

고차원 이미지 공간에서는 raw pixel에서 바로 kernel similarity를 재는 게 거칠 수 있다. 그래서 논문은 drift를 feature space로 옮긴다.

$$
\mathbb{E}\left[\| \phi(x) - \text{stopgrad}(\phi(x)+V(\phi(x))) \|^2 \right]
$$

여기서 $\phi$ 는 self-supervised encoder 같은 feature extractor다. 이건 perceptual loss와 닮아 보이지만, 중요한 차이가 있다.

- perceptual loss는 보통 특정 target sample과 pair가 필요하다.
- drifting loss는 pair가 필요 없고, **분포 수준의 positive/negative sample set** 만 있으면 된다.

즉, 이 방법은 paired supervision 없이 feature-space distribution matching을 유도한다.

### Classifier-free guidance도 training-time에 흡수

논문은 CFG도 흥미롭게 다룬다. 보통 diffusion에서는 inference 때 conditional/unconditional score를 섞어 guidance를 만든다. 여기서는 negative sample distribution을 섞는 방식으로 training-time guidance를 구현한다. 그래서 inference는 여전히 1-NFE를 유지한다.

이 부분의 메시지는 명확하다.

> **Drifting Models는 one-step property를 깨지 않으면서도 conditional guidance 구조를 흡수할 수 있다.**

## Results

결과에서 가장 중요한 숫자는 초록에 이미 나와 있다.

### 1. ImageNet 256x256에서 strong one-step generation

논문은 ImageNet 256x256에서 다음을 보고한다.

- **latent-space generation:** 1-NFE FID **1.54**
- **pixel-space generation:** 1-NFE FID **1.61**

특히 latent-space 1.54는 one-step 계열에서 SOTA라고 주장한다. 여기서 중요한 건 단지 숫자 하나가 아니라, **multi-step distillation 없이도 처음부터 one-step generator를 학습하는 새로운 경로** 를 보여준다는 점이다.

### 2. pixel-space 결과가 유난히 강하다

개인적으로 더 인상적인 건 pixel-space 1.61이다. latent tokenizer의 도움 없이도 이 정도면, 이 접근이 “latent에서만 되는 꼼수”는 아니라는 신호다. 보통 pixel-space one-step generation은 훨씬 더 어렵고 불안정한데, 이 논문은 그 간극을 꽤 많이 줄인다.

### 3. anti-symmetry는 실제로 중요하다

논문은 anti-symmetry를 일부러 깨는 destructive ablation도 보여준다. 결과는 명확하다. anti-symmetric 설계는 잘 되지만, 이를 깨면 학습이 사실상 무너진다. 이건 이론 섹션의 조건이 단순한 ornament가 아니라는 뜻이다. 실제로 **equilibrium을 제대로 만드는 구조적 제약** 으로 작동한다.

### 4. positive/negative sample 수가 많을수록 좋아진다

drifting field는 mini-batch positive/negative sample로 근사한다. 논문은 동일 compute budget 하에서 $N_{pos}$ 와 $N_{neg}$ 를 늘리면 FID가 좋아진다고 보고한다. 이것도 꽤 자연스럽다. 더 많은 sample은 더 정확한 field estimation으로 이어지기 때문이다. contrastive learning에서 batch size가 중요한 것과 비슷한 감각이다.

### 5. toy experiment가 surprisingly 설득력 있다

2D toy example에서 generated distribution이 두 모드 분포로 자연스럽게 이동하는 그림은 단순하지만 메시지가 강하다. 특히 collapsed initialization에서도 다른 mode가 계속 attract하기 때문에, mode collapse에서 빠져나올 수 있다는 직관을 준다. 물론 toy result를 과신하면 안 되지만, 이 논문의 메커니즘을 이해하는 데는 꽤 좋다.

## Discussion

내가 보기에 이 논문의 진짜 매력은 성능 숫자만이 아니라 **생성 모델의 iterative computation을 어디에 둘 것인가** 라는 질문을 재배치했다는 점이다.

우리는 너무 오랫동안 “좋은 생성은 inference에서 여러 step 돌려야 한다”는 전제를 거의 자연법칙처럼 받아들여왔다. 그런데 drifting은 이렇게 말한다.

- 반복이 꼭 필요하다면, 그 반복은 optimizer가 대신해도 된다.
- inference는 분포 진화의 결과를 읽는 한 번의 decode로 충분할 수 있다.

이 관점은 consistency, distillation, rectified flow류와도 닿아 있지만, 논문은 자신만의 독립적인 framing을 가진다. 여기서 중심은 ODE trajectory 근사가 아니라 **equilibrium-seeking field** 이다.

또 하나 좋은 점은 GAN과도 diffusion과도 완전히 같지 않다는 것이다.

- GAN처럼 one-shot generator를 쓴다.
- 하지만 adversarial min-max game에 의존하지 않는다.
- diffusion/flow처럼 distribution dynamics를 다루지만, 그 dynamics를 inference가 아니라 training에 둔다.

즉, 이 논문은 기존 축 사이 어딘가의 hybrid가 아니라, **one-step generation을 위한 세 번째 문법** 을 만들려는 시도에 가깝다.

## Limitations

물론 이 접근을 바로 “새 표준”으로 부르기엔 아직 이르다.

### 1. 이론적으로는 아직 identifiability가 완전히 닫히지 않는다

논문도 인정하듯, 일반적인 drifting field에 대해

$$
V_{p,q}=0 \Rightarrow p=q
$$

가 항상 성립하는 건 아니다. 저자들은 kernelized construction에서는 근사적으로 그럴 수 있다는 heuristic을 준다. 즉, 이론적 foundation은 흥미롭지만 아직 diffusion의 score matching만큼 정리된 상태는 아니다.

### 2. large-batch dependence가 강해 보인다

positive/negative sample 수가 중요하다는 건 장점이자 비용 신호다. distribution-level field를 잘 추정하려면 batch 구성이 중요하고, 이는 메모리/시스템 요구와 연결된다. 실전 확장성은 구현 세부에 꽤 민감할 수 있다.

### 3. feature extractor choice의 영향이 크다

feature-space drifting은 분명 강력하지만, 그만큼 어떤 encoder를 쓰느냐에 많이 의존한다. self-supervised encoder의 representation bias가 생성 품질에 얼마나 직접 개입하는지, 어디까지 일반화되는지는 더 보고 싶다.

### 4. ImageNet 중심의 검증이다

지금 결과는 인상적이지만 주로 ImageNet 256x256에 집중되어 있다. 텍스트-조건부 생성, 비디오, 3D, 오디오, 혹은 scientific domain generation에서 같은 철학이 얼마나 잘 먹히는지는 아직 열린 질문이다.

### 5. training을 inference로 옮긴 만큼, optimization burden은 커질 수 있다

이 논문의 계산은 “공짜”가 아니다. inference가 가벼워진 대신, training 중 분포 진화를 더 많이 책임져야 한다. 서비스 관점에서는 매우 매력적일 수 있지만, 학습 비용까지 포함한 total cost 관점에서는 더 넓은 비교가 필요하다.

## Conclusion

**Generative Modeling via Drifting**은 one-step generation을 위한 또 하나의 트릭 논문이라기보다, 생성 모델을 보는 시간축 자체를 바꾸는 논문이다. 분포를 inference에서 조금씩 진화시키는 대신, training 중 모델의 pushforward distribution이 drift를 따라 움직이게 만든다. 그 중심에는 anti-symmetric drifting field와 fixed-point regression이라는 surprisingly simple한 구조가 있다.

이 논문의 가장 좋은 점은 메시지가 명확하다는 것이다. 생성 모델의 품질이 꼭 iterative inference에 묶여 있을 필요는 없다. 반복은 optimizer가 담당하고, 모델은 학습이 끝난 뒤 한 번에 샘플을 뽑아도 된다. 그리고 적어도 ImageNet 256x256에서는, 그 생각이 단지 철학이 아니라 **실제로 꽤 강한 숫자** 로 이어진다.

내 기준에서 이 논문은 “diffusion 이후의 생성 모델”을 논할 때 자주 소환될 만한 아이디어다. 지금 당장 모든 생성 파이프라인이 drifting으로 갈아탈 거라고 보진 않지만, **iterative generation을 training으로 흡수하는 방향** 은 앞으로 더 자주 보게 될 것 같다.

## TL;DR

- Drifting Models는 diffusion/flow처럼 inference에서 분포를 진화시키지 않고, **training 중 pushforward distribution을 진화** 시킨다.
- 핵심 장치는 generated/data distribution 차이로부터 sample movement를 계산하는 **drifting field $V_{p,q}$** 다.
- drifting field는 **anti-symmetry** 를 만족하도록 설계되어, $p=q$ 일 때 drift가 0인 equilibrium을 만든다.
- 학습은 $x$ 를 $x + V(x)$ 라는 frozen drifted target 쪽으로 회귀시키는 **fixed-point regression objective** 로 이루어진다.
- feature-space drifting과 training-time CFG도 자연스럽게 지원한다.
- ImageNet 256x256에서 **1-NFE FID 1.54(latent)**, **1.61(pixel)** 를 기록하며 one-step generation에서 매우 강한 결과를 보인다.
- 이 방법은 GAN의 one-shot generator와 diffusion의 distribution dynamics 사이 어딘가가 아니라, **inference-time iteration을 training-time evolution으로 재배치하는 새로운 생성 문법** 에 가깝다.
- 다만 이론적 identifiability, batch-size dependence, feature extractor bias, domain generalization은 더 검증이 필요하다.

## Paper Info

- **Title:** Generative Modeling via Drifting
- **Authors:** Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He
- **Affiliations:** MIT, Harvard University
- **Venue:** arXiv preprint
- **Published:** 2026-02-06 (arXiv)
- **Paper:** https://arxiv.org/abs/2602.04770
- **Project:** https://lambertae.github.io/projects/drifting/

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다.
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
