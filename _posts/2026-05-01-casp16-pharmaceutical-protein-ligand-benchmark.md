---
title: "CASP16이 보여준 protein-ligand AI의 진짜 병목은 scoring이다"
date: 2026-05-01 16:08:00 +0900
description: "CASP16 제약 protein-ligand 벤치마크는 포즈 예측은 크게 좋아졌지만 affinity 예측은 여전히 낮은 상관에 머문다는 점을 보여주며, 현재 drug discovery AI의 핵심 병목이 scoring function에 있음을 드러낸다."
categories: [AI, Drug Discovery]
tags: [protein-ligand, docking, affinity-prediction, benchmark, casp16, drug-discovery, scoring-function, alphafold3]
math: true
mermaid: false
image:
  path: /assets/img/posts/casp16-pharmaceutical-protein-ligand-benchmark/fig1_overview.svg
  alt: "CASP16 protein-ligand benchmark summary showing strong pose prediction progress but persistent affinity scoring bottleneck"
---

## Hook

protein-ligand AI를 볼 때 자주 생기는 착시는 이거다. 구조를 그럴듯하게 잘 맞히는 모델이 나오면, 이제 binding affinity까지 곧 같이 풀릴 것처럼 느껴진다. 하지만 실제 drug discovery에서 더 어려운 질문은 늘 뒤에 남는다. **잘 놓는 것과 잘 고르는 것은 다르다.**

이번 CASP16 pharmaceutical protein-ligand assessment는 그 차이를 꽤 냉정하게 드러낸다. 블라인드 벤치마크에서 참가팀들은 small-molecule pose와 affinity를 동시에 예측해야 했고, 결과는 명확했다. **pose prediction은 꽤 좋아졌지만, affinity prediction은 여전히 scoring의 벽 앞에 있다.**

이 논문이 중요한 이유는 단순히 성능표 하나를 추가해서가 아니다. 실제 pharma discovery project에서 나온 drug-like compound를 대상으로, 현재 protein-ligand modeling이 어디까지 왔고 어디에서 막히는지를 비교적 정직하게 보여주기 때문이다. hype를 걷어내고 보면, 지금의 병목은 구조 생성보다도 **ranking과 calibration의 문제**에 더 가깝다.

## Problem

이 논문이 겨냥하는 질문은 단순하다.

> 지금의 computational protein-ligand methods는 블라인드 제약 벤치마크에서 실제로 얼마나 잘 맞히는가?

그런데 이 질문을 제대로 보려면 몇 가지 병목을 분리해서 봐야 한다.

### 병목 1: pose prediction과 affinity prediction은 같은 문제가 아니다

많은 모델이 docking pose를 그럴듯하게 만든다. 최근에는 template 활용, structure prediction 계열 모델, deep learning 기반 docking이 섞이면서 pose 자체는 빠르게 좋아졌다. 하지만 실제 drug discovery에서는 리간드를 pocket 안에 "대충 맞게 놓는 것"만으로는 부족하다.

중요한 건 보통 다음 순서다.

- 어떤 pose가 plausible한가
- 그 pose들 사이의 energetic ordering이 맞는가
- 실험 affinity와 얼마나 일관되게 rank correlation을 맞추는가

즉 pose accuracy가 좋아져도 affinity ranking이 같이 좋아진다고 보장할 수는 없다.

### 병목 2: 구조를 알아도 score가 약하면 끝까지 못 간다

이 논문에서 특히 중요한 포인트는, challenge의 두 번째 단계에서 실험 구조를 제공해도 affinity prediction이 별로 나아지지 않았다는 점이다. 이건 병목이 "구조를 못 맞혀서"라기보다, **맞힌 구조를 가지고도 좋은 score를 못 매긴다**는 뜻에 가깝다.

이 말은 곧 지금의 많은 방법이 아래 둘을 분리해서 생각해야 한다는 얘기다.

- structure generation or pose placement
- scoring, ranking, calibration

현재는 앞쪽보다 뒤쪽이 더 덜 풀려 있다.

### 병목 3: benchmark는 실제 hype filter 역할을 한다

학습 데이터셋이나 retrospective benchmark에서는 그럴듯해 보이는 방법도, blinded pharma-like benchmark에 가면 성능이 갑자기 평범해지는 경우가 많다. CASP16 같은 평가는 이런 간극을 드러내는 데 특히 유용하다.

이 논문은 결국 다음 질문을 다시 던진다.

- 모델이 정말 binding을 이해하는가
- 아니면 구조 템플릿과 dataset bias를 잘 이용하는가
- pose를 잘 내는 것과 affinity를 잘 rank하는 것 중 어느 쪽이 실제 병목인가

## Key Idea

이 논문의 핵심 기여는 새로운 생성 모델을 제안하는 데 있지 않다. 대신 **현재 방법들을 실제 제약 조건의 블라인드 평가로 압축해 보여주는 benchmark paper**라는 데 있다.

압축해서 말하면 이렇다.

- CASP16에서 참가팀들은 pose target 229개, affinity target 140개를 예측했다.
- 대상은 5개 protein system이며, 실제 pharmaceutical discovery project에서 나온 drug-like compound가 포함된다.
- pose와 affinity를 같은 challenge 안에서 나란히 평가함으로써, 현재 protein-ligand AI의 강점과 병목을 분리해 보여준다.

논문이 주는 가장 중요한 메시지는 두 줄로 줄일 수 있다.

1. **Pose prediction은 생각보다 많이 좋아졌다.**
2. **Affinity prediction은 여전히 scoring function이 약하다.**

이건 단순한 성능 비교가 아니라, 앞으로 연구 투자를 어디에 더 해야 하는지에 대한 방향 제시이기도 하다.

## How It Works

### Overview

![CASP16 benchmark overview](/assets/img/posts/casp16-pharmaceutical-protein-ligand-benchmark/fig1_overview.svg)
_Figure 1: CASP16 pharmaceutical protein-ligand benchmark의 핵심 메시지를 정리한 요약 그림. pose prediction은 크게 진전됐지만 affinity prediction은 여전히 scoring 병목이 남아 있다는 점을 강조한다._

이 논문의 "아키텍처"는 모델 아키텍처라기보다 **평가 파이프라인 아키텍처**에 가깝다. 즉, 어떤 입력을 주고, 어떤 단계로 평가하고, 무엇을 metric으로 읽는지가 핵심이다.

전체 흐름은 대략 아래처럼 이해하면 된다.

1. 제약 discovery project에서 유래한 protein-ligand target을 준비한다.
2. 참가팀들은 각 target에 대해 binding pose를 예측한다.
3. 별도 affinity target에 대해 binding strength ordering 또는 value를 예측한다.
4. pose는 structural quality metric으로, affinity는 rank correlation 중심으로 평가한다.
5. 같은 문제에 대해 baseline automated methods도 함께 돌려 비교한다.

즉 이 논문은 "최신 모델이 얼마나 멋진가"보다,

- 블라인드 조건에서
- pose와 affinity를 분리해서 보면
- 실제로 어디가 강하고 어디가 약한가

를 보는 평가 시스템이다.

### Representation / Formulation

이 벤치마크의 핵심은 pose quality와 affinity quality를 다른 종류의 관측량으로 본다는 점이다.

Pose prediction 쪽은 구조적 정합성을 보고, affinity prediction 쪽은 실험값과의 순위 일치 정도를 본다. 논문 abstract 기준 핵심 수치는 아래 둘이다.

$$
\text{Best mean LDDT-PLI for submitted pose predictors} = 0.69
$$

$$
\text{Best Kendall's } \tau \text{ for affinity prediction} < 0.42
$$

여기서 메시지는 꽤 선명하다.

- pose metric은 상위권에서 이미 꽤 높다
- affinity correlation은 여전히 modest한 수준이다

또 논문은 experimental uncertainty를 고려한 affinity prediction의 이론적 상한을 대략 다음 수준으로 본다.

$$
\tau_{\max,\ theoretical} \approx 0.73
$$

즉 현재 최고 성능과 이론적 ceiling 사이에도 아직 꽤 간극이 남아 있다.

### Core Evaluation Pipeline

이 논문을 읽을 때 중요한 건 참가 방법의 세세한 구현보다, benchmark가 무엇을 드러내는 구조인지다.

- **입력**
  - 단백질 target
  - 소분자 ligand
  - challenge stage에 따라 구조 정보 수준이 다름
- **계산**
  - pose generation or docking
  - affinity scoring or ranking
- **출력**
  - predicted complex pose
  - predicted affinity ordering or values
- **평가**
  - pose: structural correctness
  - affinity: experimental measurement와의 correlation

재미있는 건 pose 쪽에서는 template-based 방법과 자동 baseline이 강하게 보였고, affinity 쪽에서는 그런 구조적 이점이 그대로 이어지지 않았다는 점이다.

간단히 적으면 현재 파이프라인은 아래처럼 볼 수 있다.

```python
import torch
import torch.nn as nn

class ProteinLigandWorkflow(nn.Module):
    def __init__(self, pose_model, scoring_function):
        super().__init__()
        self.pose_model = pose_model
        self.scoring_function = scoring_function

    def forward(self, protein, ligand):
        pose = self.pose_model(protein, ligand)
        affinity = self.scoring_function(protein, ligand, pose)
        return pose, affinity
```

이 워크플로에서 논문이 보여주는 병목은 대체로 아래 위치에 있다.

- `pose_model(...)`은 꽤 강해지고 있음
- `scoring_function(...)`은 아직 실험 affinity를 안정적으로 설명하지 못함

즉 병목은 구조 생성 엔진보다 score layer 쪽에 더 가깝다.

### Training / Inference Pipeline

이 benchmark paper를 실무 관점에서 다시 쓰면, 현재 field의 전형적 inference 흐름은 대략 이런 느낌이다.

```text
Input protein target and ligand candidates
→ generate candidate poses with template-based, docking, or DL models
→ optionally refine using provided or predicted structures
→ score / rank candidates
→ compare predicted ordering with experimental affinity
```

여기서 논문이 말해주는 중요한 사실은 다음이다.

- 더 좋은 구조 정보를 줘도
- affinity ranking이 거의 안 오를 수 있다
- 그러면 병목은 sampling보다 scoring일 가능성이 크다

즉 연구 방향도 자연스럽게 바뀐다.

- 더 좋은 pose generator 만들기
- 더 나은 physics-aware or ML scoring 만들기
- assay noise와 label uncertainty를 모델링하기
- target별 calibration을 더 잘하기

### Stage 1 vs Stage 2가 왜 중요했나

이 논문에서 내가 특히 중요하게 보는 부분은 challenge를 사실상 두 번의 질문으로 나눠 읽을 수 있다는 점이다.

- **Stage 1 성격의 질문**
  - 구조 정보가 제한된 상태에서 pose와 affinity를 얼마나 맞히는가
- **Stage 2 성격의 질문**
  - 더 좋은 구조 정보, 심지어 experimental structure에 가까운 정보를 줬을 때 affinity가 얼마나 개선되는가

직관적으로는 Stage 2에서 성능이 꽤 올라야 할 것처럼 느껴진다. 구조 오차가 줄어들면 그 위에서 더 정확한 interaction reasoning이 가능할 것 같기 때문이다. 그런데 논문은 그 기대가 크게 성립하지 않았다고 말한다. 이건 아래 해석으로 이어진다.

- pose uncertainty는 물론 문제다
- 하지만 그것만 줄인다고 affinity 문제가 자동으로 풀리진 않는다
- 결국 score landscape 자체가 충분히 정렬되지 않았을 수 있다

실무적으로 이건 꽤 큰 의미가 있다. 예를 들어 docking pipeline에서 expensive refinement를 더 넣는 것이 실제 ranking 개선으로 이어질지, 아니면 그냥 더 비싼 구조만 얻고 끝날지를 가르는 힌트가 되기 때문이다.

### Why this works

이 benchmark가 특히 설득력 있는 이유는 두 가지다.

첫째, 문제를 실제 pharma-like setting으로 가져왔기 때문이다. synthetic toy benchmark보다 훨씬 현실적인 질문을 던진다.

둘째, pose와 affinity를 한 challenge 안에서 분리해 보여주기 때문이다. 이 덕분에 "모델이 전반적으로 좋아졌다"는 모호한 평가 대신, 정확히 어디가 좋아졌고 어디가 아닌지를 볼 수 있다.

그 결과, 지금의 protein-ligand AI에 대해 꽤 현실적인 결론을 내릴 수 있다.

- bound pose generation은 이미 꽤 경쟁력 있다
- 하지만 medicinal chemistry 관점에서 더 중요한 ranking 문제는 여전히 어렵다
- 따라서 다음 단계의 핵심은 better structure alone이 아니라 better scoring이다

## Results

논문 abstract와 메타데이터에서 바로 읽히는 핵심 결과는 다음과 같다.

- 참가 그룹 수: **30개 연구 그룹**
- pose target: **229개**
- affinity target: **140개**
- protein system: **5개**
- 최고 제출 그룹의 평균 pose 성능: **LDDT-PLI 0.69**
- AlphaFold 3 automated baseline: **mean LDDT-PLI 0.80**
- 최고 affinity correlation: **Kendall's \(\tau < 0.42\)**
- experimental uncertainty를 고려한 이론적 상한: **약 0.73**

이걸 해석하면 이렇다.

### Pose 쪽 해석

상위권 pose 성능은 이미 꽤 높고, 자동 baseline 중 AlphaFold 3가 참가팀 최고 성능보다 높게 나온 점은 인상적이다. 적어도 구조적으로 그럴듯한 complex pose를 제안하는 능력은 최근 foundation model 계열과 template-aware 방법 덕분에 크게 올라왔다고 볼 수 있다.

### Affinity 쪽 해석

반면 affinity는 여전히 modest correlation 수준이다. 더 중요한 건 experimental structure를 준 뒤에도 개선이 제한적이었다는 점이다. 이건 정확한 구조를 아는 것만으로는 충분하지 않고, 그 위에서 어떤 energetic proxy를 학습하거나 계산하느냐가 더 중요하다는 뜻이다.

### Benchmark positioning

논문은 overall accuracy가 prior D3R blinded prediction challenges와 비슷한 수준이라고 정리한다. 즉 field가 완전히 정체됐다는 뜻은 아니지만, 최소한 affinity prediction에서 혁명적인 도약이 이미 일어났다고 보기도 어렵다.

## Discussion

이 논문은 protein-ligand AI를 보는 관점을 조금 교정해준다.

최근 몇 년간 구조 생성 모델은 정말 빠르게 발전했다. co-folding, template retrieval, all-atom interaction modeling이 섞이면서 "복합체 모양을 잘 만드는 능력" 자체는 눈에 띄게 좋아졌다. 그런데 실제 medicinal chemistry workflow에서 더 비싼 결정은 대개 ranking에서 나온다.

- 어떤 화합물을 먼저 합성할지
- 어떤 scaffold를 밀지
- 어떤 series를 버릴지
- 어느 hit를 lead optimization으로 넘길지

이 결정에는 pose snapshot보다도 **상대적 affinity ordering**이 더 중요하다. CASP16 결과는 바로 이 지점에서 아직 큰 간극이 남아 있음을 보여준다.

내가 보기엔 이 논문이 특히 중요한 이유는, 지금의 field가 다음 착각에서 벗어나게 해주기 때문이다.

> 구조 모델이 좋아지면 drug discovery score도 자동으로 따라 좋아질 것이다.

현실은 그보다 훨씬 덜 자동적이다. 구조를 잘 그리는 모델과, binding energetics를 잘 반영하는 모델은 아직 같은 것이 아니다.

## Limitations

이 논문도 물론 몇 가지 한계가 있다.

- **benchmark scope가 제한적이다**
  - 5개 protein system, 229 pose target, 140 affinity target은 의미 있지만 여전히 전 우주를 대표하진 않는다.
- **abstract 기준으로는 방법별 세부 실패 모드가 충분히 드러나지 않는다**
  - 어떤 target family에서 특히 약했는지, pose error와 affinity error가 어떻게 연결되는지는 본문 전체를 더 세밀하게 봐야 한다.
- **AlphaFold 3 같은 강한 baseline의 practical cost는 별도로 봐야 한다**
  - accuracy와 throughput, 데이터 접근성, 실제 배치 운영 가능성은 같은 문제가 아니다.
- **affinity metric은 assay noise와 experimental setup의 영향도 받는다**
  - 따라서 낮은 correlation을 전부 모델 탓으로만 돌릴 수는 없다.

그래도 이런 한계를 감안해도, scoring bottleneck이 남아 있다는 큰 방향성 자체는 충분히 설득력 있다.

## Conclusion

CASP16 pharmaceutical protein-ligand benchmark는 현재 protein-ligand AI의 실력을 꽤 솔직하게 보여준다. pose prediction은 template-based 접근과 최신 구조 모델 덕분에 상당한 수준까지 올라왔고, automated baseline인 AlphaFold 3도 강한 성능을 보였다. 하지만 affinity prediction은 여전히 낮은 correlation에 머물렀고, experimental structure를 제공해도 큰 개선이 없었다.

결국 이 논문이 주는 핵심 메시지는 분명하다.

> **이제 병목은 pose generation만이 아니라 scoring이다.**

앞으로 진짜 어려운 문제는 구조를 더 예쁘게 만드는 것보다, 그 구조 위에서 실제 binding strength를 더 잘 설명하고 더 잘 rank하는 것이다.

## TL;DR

- CASP16은 pose와 affinity를 함께 본 pharma-like blinded benchmark라서 현실성이 높다.
- pose prediction은 이미 꽤 강해졌고, AlphaFold 3 baseline도 매우 높은 성능을 보였다.
- 하지만 affinity prediction은 여전히 modest correlation에 머물러, 현재 protein-ligand AI의 핵심 병목이 scoring function에 있음을 보여준다.

## Paper Info

- **Title:** Assessment of Pharmaceutical Protein-Ligand Pose and Affinity Predictions in CASP16
- **Authors:** Michael K. Gilson, Jerome Eberhardt, Peter Škrinjar, Janani Durairaj, Xavier Robin, Andriy Kryshtafovych
- **Affiliations:** University of California San Diego, SIB Swiss Institute of Bioinformatics, University of Basel, University of California Davis
- **Venue:** Proteins
- **Published:** 2025-10-04 online, 2026-01 print issue
- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12750038/
- **Project:** N/A
- **Code:** N/A
