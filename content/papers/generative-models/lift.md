---
title: LiFT — Language-Informed Flow Matching for Trend-Guided Structure-Based 3D Molecular Generation
aliases:
  - papers/lift
  - papers/generative-models/lift
  - papers/sbdd/lift
tags:
  - papers
  - generative-models
  - flow-matching
  - molecular-generation
  - conditional-generation
  - structure-based-modeling
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2608.31009
---

# LiFT: Language-Informed Flow Matching for Trend-Guided Structure-Based 3D Molecular Generation

> **한 줄 요약:** LiFT의 핵심은 LLM이 3D molecule을 직접 만드는 것이 아니라, language-derived chemical prior를 **Flow Matching velocity field의 조건**으로 제공하고, 현재 생성 중인 3D state를 보면서 그 조건의 영향력을 동적으로 조절하는 것입니다.

## 왜 이 논문을 저장하는가

3D molecular generation에서 자주 부딪히는 문제는 두 종류의 조건이 서로 다른 표현 공간에 있다는 점입니다.

- pocket geometry와 steric compatibility는 **3D geometric constraint**입니다.
- QED, synthetic accessibility, scaffold preference, chemical trend는 대체로 **1D/2D chemical-semantic constraint**입니다.

한 모델에 조건을 계속 추가하면 두 정보가 충돌할 수 있습니다. 반대로 chemistry condition마다 generator를 다시 fine-tune하면 유지비가 커집니다.

LiFT가 흥미로운 이유는 이 문제를 다음처럼 분업하기 때문입니다.

$$
\text{language / chemical model}
\rightarrow
\text{semantic trend}
\rightarrow
\text{state-dependent routing}
\rightarrow
\text{3D geometric flow}.
$$

즉 language model은 **어떤 방향의 molecule을 선호할지**를 제안하고, geometric generator가 **그 조건을 3D pocket 안에서 실현할 수 있는지**를 책임집니다. 이 역할 분리가 논문의 가장 재사용 가능한 아이디어입니다.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Language-Informed Flow Matching for Trend-Guided Structure-Based 3D Molecular Generation |
| Authors | Tianyu Gao, Zhikai Su, Jiashu Li, Wenjun Gao, Zichuan Ying, Zhe Zhao, Fei Zhang, Ye Wei |
| Year | 2026 |
| Venue | Findings of EMNLP 2026 |
| arXiv | [2608.31009](https://arxiv.org/abs/2608.31009) |
| Code | [kasurl/LiFT](https://github.com/kasurl/LiFT) |
| Backbone | DrugFlow-style pocket-conditioned 3D Flow Matching |
| Main evaluation | CrossDocked2020 |
| Public artifacts | checkpoint, condition files / embeddings, processed CrossDocked data |

## 1. Problem: geometry와 chemistry preference를 동시에 다루기

Structure-based molecular generation은 단순히 valid SMILES를 만드는 문제가 아닙니다. 최종 sample은 최소한 다음 두 축을 동시에 만족해야 합니다.

$$
\underbrace{\text{pocket-compatible 3D geometry}}_{\text{where / how it fits}}
\qquad + \qquad
\underbrace{\text{chemical preference}}_{\text{what kind of molecule we want}}.
$$

기존 controllable generation은 흔히 다음 중 하나를 사용합니다.

1. 특정 property를 위해 generator를 다시 fine-tune합니다.
2. sampling 중 external score나 gradient를 넣습니다.
3. 조건 자체를 generator representation에 강하게 concatenate합니다.

각 방법에는 비용이 있습니다.

- task별 fine-tuning은 새로운 objective가 추가될 때마다 모델 관리 비용이 생깁니다.
- sampling-time guidance는 현재 evolving geometry와 외부 objective가 충돌할 수 있습니다.
- static conditioning은 ODE trajectory의 모든 단계에서 같은 강도로 조건을 적용합니다.

마지막 문제가 특히 중요합니다. 3D generation 초반의 noisy state와 후반의 거의 완성된 molecular state가 같은 종류의 semantic intervention을 필요로 한다고 볼 이유는 없습니다.

LiFT의 질문은 따라서 다음처럼 읽는 것이 좋습니다.

> **semantic condition을 3D generator에 넣되, 현재 geometric state에 따라 condition의 영향력을 달리할 수 있는가?**

## 2. 전체 pipeline

공식 구현이 설명하는 기본 흐름은 다음과 같습니다.

```text
protein pocket + task preference (+ optional reference molecule)
    ↓
Sense–Evolve–Assemble agent
    ↓
target-aware SMILES conditions
    ↓
SMILES validation / repair
    ↓
frozen SMI-TED molecular-language encoder
    ↓
continuous semantic prior
    ↓
semantic projector + SCDR
    ↓
DrugFlow-style pocket-conditioned 3D Flow Matching
    ↓
3D ligand candidates
```

여기서 가장 중요한 해석은:

$$
\text{LLM output} \neq \text{final molecule}.
$$

LLM/agent가 만든 SMILES는 **intermediate condition**입니다. 최종 3D coordinates와 atom/bond state는 geometric generator가 생성합니다.

이 구조는 `LLM as molecule generator`와 상당히 다릅니다. Language model이 직접 3D coordinate validity를 책임지지 않으므로, semantic knowledge와 geometric inductive bias를 분리할 수 있습니다.

## 3. Sense–Evolve–Assemble: agent는 무엇을 하는가

LiFT의 agent stage는 target pocket과 task preference를 읽어 **target-aware chemical condition**을 만드는 역할을 합니다.

개념적으로는:

$$
(p, \tau, r)
\xrightarrow{\text{agent}}
s,
$$

where

- $p$: pocket/context description,
- $\tau$: 원하는 chemical trend 또는 task preference,
- $r$: optional reference molecule,
- $s$: intermediate SMILES condition.

이 단계의 중요한 boundary는 agent가 구조적 정답을 생성하는 것이 아니라는 점입니다.

agent가 할 일:

- pocket context를 chemistry language로 요약,
- 원하는 property trend를 표현,
- 필요하면 reference scaffold를 반영,
- downstream encoder가 읽을 수 있는 chemical condition을 제안.

agent가 직접 보장하지 않는 것:

- pocket pose correctness,
- exact affinity,
- physical clash removal,
- atom-level geometric feasibility.

따라서 LiFT를 평가할 때는 `agent가 좋은 SMILES를 만들었다`와 `그 semantic condition이 3D generator를 실제로 개선했다`를 분리해야 합니다.

## 4. Semantic prior: discrete SMILES를 continuous condition으로 바꾸기

Agent가 출력한 SMILES를 그대로 geometric model에 넣는 대신, LiFT는 frozen molecular-language encoder를 통해 continuous semantic vector로 바꿉니다.

$$
s
\xrightarrow{E_{\text{chem}}}
z_{\text{sem}}.
$$

공식 구현은 SMI-TED 기반의 768-dimensional representation을 사용합니다. Released precomputed embedding을 사용하면 SMI-TED 자체를 inference 환경에 설치하지 않아도 됩니다.

이 설계에는 두 가지 의미가 있습니다.

### 4.1 조건을 geometry와 분리할 수 있다

SMILES condition은 chemical identity/trend를 담지만 좌표계 정보를 담지 않습니다. 따라서 semantic pathway가 rotation-dependent vector를 직접 만들어 geometric state에 섞는 것보다 symmetry 관리가 쉽습니다.

### 4.2 foundation model을 generator로 다시 학습하지 않는다

Chemical encoder는 frozen prior 역할을 합니다. LiFT에서 학습되는 것은 이 prior를 3D generator의 dynamics로 **어떻게 연결할지**입니다.

즉 핵심 문제는:

$$
\text{semantic knowledge acquisition}
\quad\text{보다}\quad
\text{semantic-to-geometry interface learning}
$$

에 가깝습니다.

## 5. Flow Matching 관점에서 보기

Flow Matching은 초기 distribution의 sample을 target data distribution으로 옮기는 continuous dynamics를 학습합니다.

추상적으로:

$$
\frac{d x_t}{dt} = v_\theta(x_t,t,c),
$$

where

- $x_t$: intermediate molecular state,
- $t$: flow time,
- $c$: pocket 및 condition,
- $v_\theta$: learned velocity field.

LiFT는 generator의 transition family를 완전히 바꾸기보다 condition $c$에 language-derived semantic prior를 추가합니다.

$$
v_\theta
=
v_\theta(x_t,t,p,z_{\text{sem}}).
$$

하지만 여기서 단순히 $z_{\text{sem}}$을 모든 layer와 모든 time step에 같은 방식으로 주입하면 문제가 남습니다.

초반 state:

$$
x_t \approx \text{poorly organized / noisy geometry}
$$

후반 state:

$$
x_t \approx \text{chemically and geometrically structured candidate}
$$

이 둘의 condition 필요량이 같다는 보장이 없습니다.

그래서 SCDR가 등장합니다.

## 6. SCDR: state-dependent semantic routing

**Self-Conditioned Decoupled Router (SCDR)**는 현재 ODE intermediate state를 읽고 semantic contribution을 조절합니다.

가장 간단한 추상화는:

$$
g_t = R_\phi(x_t,t,z_{\text{sem}}),
$$

$$
v'_\theta(x_t,t)
=
v_\theta(x_t,t,p)
+
g_t \odot \Delta v_\theta(x_t,t,z_{\text{sem}}).
$$

$g_t$가 중요한 이유는 semantic condition이 **static context**가 아니라 trajectory-dependent intervention으로 바뀌기 때문입니다.

### 왜 self-conditioned인가

Router는 오직 semantic vector와 time만 보는 것이 아니라, 현재 생성 중인 molecular state의 summary를 봅니다.

따라서:

$$
\text{same semantic preference}
+
\text{different intermediate structure}
\Rightarrow
\text{different routing strength}.
$$

이 점이 LiFT의 가장 재사용 가능한 architecture idea입니다.

## 7. Equivariance를 깨지 않고 semantic prior를 넣는 방법

3D molecular generation에서 condition injection이 까다로운 이유는 geometric representation의 transformation rule 때문입니다.

좌표가 rotation $R$과 translation $t$를 받으면:

$$
x_i' = R x_i + t.
$$

vector feature $h_i^{(1)}$는 일반적으로:

$$
h_i^{(1)\prime}=Rh_i^{(1)}
$$

처럼 변해야 하고, invariant scalar는 변하지 않아야 합니다.

LiFT의 설계에서 중요한 점은 language-derived semantic information을 **invariant/scalar path**를 통해 geometric network에 주입하는 것입니다. 현재 vector state의 방향 성분 자체를 language embedding으로 직접 덮어쓰는 방식이 아닙니다.

SCDR가 geometric state를 요약할 때도 vector의 norm처럼 invariant quantity를 사용할 수 있습니다.

$$
\|Rh_i\|_2 = \|h_i\|_2.
$$

따라서 router가 invariant summary에서 scalar gate를 만들고 이 gate를 equivariant vector update에 broadcast한다면:

$$
g(Rx)=g(x)
$$

이고

$$
g(x)\,Rh = R(g(x)h),
$$

이므로 기존 equivariance contract를 유지할 수 있습니다.

이것은 cross-modal model을 설계할 때 매우 일반적인 원칙입니다.

> **외부 semantic modality가 geometric transformation law를 알 필요가 없도록, invariant interface를 만든다.**

## 8. Zero-initialized conditioning은 왜 필요한가

새 condition path를 pretrained geometric generator에 추가할 때 가장 쉬운 실패는 초기부터 semantic branch가 너무 강하게 작동하는 것입니다.

LiFT는 zero-initialized adaptive normalization 계열의 interface를 사용해 시작 시점의 behavior가 기존 generator와 가까워지도록 만듭니다.

개념적으로:

$$
h' = \operatorname{Norm}(h)\odot(1+\gamma(z)) + \beta(z),
$$

초기에는

$$
\gamma(z)\approx 0,\qquad \beta(z)\approx 0.
$$

그러면 학습 초기에는:

$$
h' \approx \operatorname{Norm}(h),
$$

이고 semantic pathway는 필요한 만큼만 점진적으로 영향력을 얻습니다.

이 장치는 단순 training trick이 아니라 **기존 geometric prior를 보존하면서 새로운 modality를 붙이는 interface contract**로 이해하는 편이 좋습니다.

## 9. Training / inference contract

공식 repository의 released training configuration은 다음과 같은 규모를 명시합니다.

- heterogeneous GVP layers: 5
- batch size: 48
- learning rate: $8\times10^{-4}$
- training epochs: 500
- training flow steps: 5,000
- sampling steps: 500

Paper checkpoint는 epoch 399로 공개되어 있습니다.

기본 sampling configuration은 released **Balanced-NoRef** semantic embedding을 사용하고, test target당 100 samples를 생성합니다.

이 부분은 reproducibility 측면에서 꽤 좋은 편입니다. Repository에 다음이 따로 존재합니다.

- training config,
- sampling config,
- condition-generation scripts,
- embedding-generation scripts,
- ablations,
- validation scripts,
- evaluation / post-processing scripts.

또한 checkpoint와 condition archive가 공개되어 있어 `논문 아이디어만 있고 재현 경로가 없는` 유형은 아닙니다.

## 10. Evaluation contract를 먼저 읽어야 한다

LiFT의 main benchmark는 refined CrossDocked2020입니다. 여기서 중요한 것은 하나의 metric으로 논문 전체를 판단하지 않는 것입니다.

3D generation의 metric들은 서로 다른 claim을 측정합니다.

| Metric family | 실제 질문 |
| --- | --- |
| QED / SA | drug-like chemical preference가 좋아졌는가? |
| LogP / ring statistics | 원하는 chemical distribution/trend를 따르는가? |
| Vina / Gnina | docking/scoring proxy가 개선되는가? |
| PoseBusters | 생성된 3D structure가 물리적으로 그럴듯한가? |
| RDKit / REOS filters | 기본 chemistry/filter compliance가 유지되는가? |
| Wasserstein distance | generated property distribution이 reference distribution과 얼마나 비슷한가? |

따라서:

$$
\text{higher QED}
\not\Rightarrow
\text{better binding affinity}
$$

이고

$$
\text{better docking proxy}
\not\Rightarrow
\text{prospective biological activity}.
$$

이 boundary를 유지해야 LiFT의 claim을 과도하게 확장하지 않을 수 있습니다.

## 11. 핵심 결과를 어떻게 읽을 것인가

논문의 no-reference steering setting에서 몇 가지 대표 결과를 보면 chemical preference와 3D validity 사이의 trade-off가 드러납니다.

| Model / condition | QED ↑ | SA ↓ | RDKit ↑ | REOS ↑ | PoseBusters ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DrugFlow baseline | 0.553 | 3.43 | 75.86 | 64.84 | 78.45 |
| LiFT QED-NoRef | 0.744 | 2.724 | 83.19 | 71.10 | 72.21 |
| LiFT Vina-NoRef | 0.732 | 2.662 | 83.36 | 73.81 | 73.56 |
| LiFT Balanced-NoRef | **0.757** | **2.659** | 81.27 | 71.78 | 70.73 |

여기서 가장 중요한 점은 `LiFT가 모든 metric에서 baseline을 이긴다`가 아닙니다.

오히려 표는 다음을 보여줍니다.

1. QED/SA와 chemistry filter compliance는 강하게 개선됩니다.
2. PoseBusters는 DrugFlow baseline보다 낮아집니다.
3. 즉 semantic steering에는 실제 trade-off가 있습니다.
4. 논문의 좋은 읽기는 **trade-off를 없앴다**가 아니라 **generator fine-tuning 없이 trade-off frontier를 조절할 수 있다**에 가깝습니다.

이 차이는 중요합니다.

## 12. Ablation: 무엇이 실제로 필요한가

Long-form review에서 가장 중요하게 볼 부분은 전체 score보다 ablation입니다.

논문은 condition interface를 제거하거나 단순화한 설정들을 비교합니다.

대표적으로:

- ligand/reference embedding만 사용하는 경우,
- zero-initialized interface를 제거하는 경우,
- SCDR 없이 semantic information을 단순 주입하는 경우.

보고된 ablation에서는 `w/o SCDR`가 QED와 filter compliance에서 상당한 하락을 보이며, 단순 semantic injection이 full router를 대체하지 못합니다.

대략적인 패턴은:

| Ablation | QED | SA | RDKit | REOS |
| --- | ---: | ---: | ---: | ---: |
| Ligand-Ref embedding | 0.480 | 3.660 | 59.69 | 47.14 |
| w/o Zero-Init | 0.470 | 3.660 | 58.00 | 47.00 |
| w/o SCDR | 0.464 | 3.591 | 57.28 | 44.62 |

따라서 paper의 central architecture claim은 단순히:

> chemical embedding을 넣으면 좋아진다

가 아니라 다음에 더 가깝습니다.

> **semantic condition을 geometric trajectory의 상태에 맞춰 routing하는 interface가 중요하다.**

다만 이 ablation만으로 SCDR의 모든 설계 선택이 최적이라고 말할 수는 없습니다. 더 단순한 learned time-dependent gate, FiLM, cross-attention, invariant hypernetwork와 같은 controls가 필요합니다.

## 13. Agent contribution은 별도로 봐야 한다

LiFT에는 두 종류의 novelty가 동시에 들어 있습니다.

1. target-aware chemical condition을 만드는 agent pipeline,
2. condition을 geometric flow에 전달하는 state-dependent router.

따라서 전체 LiFT가 좋아졌다는 사실만으로 agent가 필요한지 알 수 없습니다.

가장 중요한 control은 다음과 같습니다.

```text
A. no semantic condition
B. manually / property-derived semantic condition
C. direct LLM condition
D. full Sense–Evolve–Assemble agent condition
```

논문 appendix의 condition-source experiments는 direct generation, retrieval, agent variants 등을 비교하며 full agent가 여러 chemical/interaction metric 사이에서 더 균형 잡힌 trade-off를 보인다고 보고합니다.

하지만 여기서도 해석을 좁혀야 합니다.

- agent가 target-specific physical reasoning을 완전히 학습했다는 증거는 아닙니다.
- prompt/LLM family가 가진 medicinal-chemistry prior가 기여할 수 있습니다.
- target pocket의 information과 reference condition의 information budget이 정확히 같은지 확인해야 합니다.

따라서 agent stage를 평가할 때는 **정보 접근량을 맞춘 control**이 필요합니다.

## 14. Trend-level control이라는 표현이 정확하다

LiFT authors가 `trend-guided`라는 표현을 쓰는 것은 적절합니다.

Semantic condition은 exact molecule을 지정하는 hard constraint가 아닙니다. Pocket ensemble 수준에서 condition과 generated property가 방향성 있게 연동되는지를 봅니다.

논문은 여러 property에서 condition과 generated ensemble의 trend correspondence를 보고합니다. 이 결과는 semantic condition이 무시되지 않는다는 evidence입니다.

하지만 이것은 다음을 뜻하지 않습니다.

$$
\text{desired QED}=0.80
\Rightarrow
\text{every generated molecule QED}=0.80.
$$

보다 현실적인 claim은:

$$
\Delta z_{\text{condition}}
\Rightarrow
\mathbb{E}[\text{generated property}]\text{가 같은 방향으로 이동}.
$$

즉 exact control이 아니라 **distribution-level steering**입니다.

## 15. What is actually new?

LiFT의 novelty를 구성 요소별로 분리하면 다음과 같습니다.

### Language agent 자체

LLM이 molecule-related condition을 만드는 것 자체는 완전히 새로운 문제 설정은 아닙니다.

### Chemical foundation-model embedding

SMILES를 pretrained chemical representation으로 바꾸는 것도 단독으로는 강한 novelty가 아닙니다.

### Flow Matching backbone

DrugFlow 계열 pocket-conditioned flow backbone이 있으므로 Flow Matching 자체가 novelty는 아닙니다.

### 진짜 핵심

가장 강한 architecture contribution은:

$$
\boxed{\text{semantic prior} + \text{current 3D state} \rightarrow \text{dynamic routing of geometric dynamics}}
$$

입니다.

그리고 두 번째로 중요한 contribution은 외부 semantic modality를 scalar/invariant interface로 제한해 geometric symmetry를 망가뜨리지 않는 cross-modal design입니다.

따라서 이 논문을 기억할 때 `LLM + molecule generation`보다 **state-dependent cross-modal conditioning**으로 기억하는 편이 더 정확합니다.

## 16. 이 논문이 잘 보여주는 것

### 16.1 semantic prior가 3D generator를 움직일 수 있다

Condition이 generated distribution의 chemical properties를 의미 있게 이동시킵니다.

### 16.2 generator를 condition마다 다시 fine-tune할 필요는 없다

동일 generator에서 condition을 바꿔 여러 steering objective를 표현할 수 있습니다.

### 16.3 state-dependent router가 단순 condition injection보다 낫다

Ablation은 full routing interface가 중요한 component임을 지지합니다.

### 16.4 공개 artifact가 비교적 충분하다

Checkpoint, configs, condition files/embeddings, processed data와 validation/evaluation scripts가 공개되어 있어 reproduction path가 비교적 명확합니다.

## 17. 이 논문이 아직 증명하지 못한 것

### 17.1 broad pocket OOD generalization

Main evidence가 CrossDocked2020 중심이므로 completely novel target families나 강한 pocket-distribution shift에 대한 일반화는 별도 검증이 필요합니다.

특히 random/standard test pocket performance와:

- protein-family split,
- pocket-similarity split,
- temporal split,
- ligand-scaffold split

은 서로 다른 주장입니다.

### 17.2 실제 binding affinity 개선

Vina/Gnina score는 useful proxy이지만 experimental affinity가 아닙니다.

$$
\text{docking score}
\neq
K_d / K_i / IC_{50}.
$$

### 17.3 biological dynamics

Static pocket generation은 induced fit, water network, alternative conformations, protein dynamics를 해결했다는 뜻이 아닙니다.

### 17.4 language reasoning이 반드시 필요한가

좋은 chemical encoder + task vector + dynamic router만으로 비슷한 효과를 낼 수 있는지 아직 중요한 질문으로 남습니다.

## 18. 가장 중요한 추가 ablation

내가 이 논문을 재현한다면 가장 먼저 다음 4-arm experiment를 돌립니다.

| Arm | Semantic input | Router |
| --- | --- | --- |
| A | none | none |
| B | fixed embedding | static |
| C | fixed embedding | time-only learned gate |
| D | fixed embedding | state-dependent SCDR |

여기서 semantic source를 고정하는 이유는 agent quality와 routing quality를 분리하기 위해서입니다.

측정할 것은:

- QED / SA / LogP,
- RDKit / REOS validity,
- PoseBusters,
- Vina / Gnina,
- diversity / novelty,
- scaffold distribution,
- pocket-family OOD,
- ligand-scaffold OOD,
- ODE trajectory 중 clash 또는 instability,
- NFE / wall time / memory.

### 결정 기준

SCDR의 architecture claim이 강해지려면 다음이 필요합니다.

$$
D > C > B > A
$$

가 단지 QED 하나가 아니라 **controllability–validity frontier**에서 나타나야 합니다.

반대로 C와 D가 거의 같다면 핵심은 `state-dependent routing`이 아니라 단순히 `learned schedule`일 수 있습니다.

## 19. 더 강한 generalization test

CrossDocked ID 성능 이후에는 split을 의도적으로 어렵게 해야 합니다.

### Pocket OOD

protein sequence 또는 structure similarity로 cluster를 만들고 cluster 단위로 split합니다.

### Ligand scaffold OOD

Bemis–Murcko scaffold가 train/test를 넘지 않게 합니다.

### Joint OOD

가장 강한 형태는:

$$
\text{new protein family}
+
\text{new ligand scaffold}.
$$

여기서 semantic prior가 여전히 도움이 된다면 `benchmark-specific chemistry bias`보다 더 강한 evidence가 됩니다.

## 20. Reproducibility checklist

공식 repository 기준으로 시작할 때 다음 순서가 합리적입니다.

1. Released checkpoint를 validation script로 확인합니다.
2. Released Balanced-NoRef condition embedding으로 sampling을 재현합니다.
3. 동일 evaluation scripts로 baseline metric을 확인합니다.
4. SCDR path와 semantic projector를 분리합니다.
5. no/static/time-only/state-dependent ablation을 같은 seed와 sampling budget에서 비교합니다.
6. CrossDocked2020 이후 별도 OOD split을 추가합니다.

주의할 점:

- SMI-TED weights는 repository 내부에 재배포되지 않습니다.
- released embedding을 사용하면 SMI-TED dependency를 피할 수 있습니다.
- GNINA-based evaluation은 별도 executable이 필요합니다.
- training과 sampling config의 일부 sampling-related setting이 다를 수 있으므로 exact released config를 기록해야 합니다.

## 21. Failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| QED 상승을 binding improvement로 해석 | chemical preference와 target affinity는 다른 claim |
| standard CrossDocked test를 broad OOD로 해석 | pocket/scaffold leakage 가능성 |
| agent 효과와 router 효과를 합쳐 평가 | 어느 component가 gain을 만드는지 알 수 없음 |
| semantic condition을 너무 강하게 적용 | pocket geometry/3D validity를 희생할 수 있음 |
| PoseBusters 하락을 숨기고 chemistry metric만 보고 | 실제 trade-off frontier를 오해 |
| docking score만으로 prospective utility 주장 | scoring-function bias와 experimental gap |
| reference molecule 정보량을 맞추지 않은 비교 | scaffold-hopping arm에 정보 advantage가 생길 수 있음 |
| LLM family 하나의 결과를 일반적 language prior로 해석 | model-specific chemistry bias 가능성 |

## 22. 다른 개념과의 연결

### [[concepts/generative-models/flow-matching|Flow Matching]]

LiFT의 geometric dynamics를 이해하는 기본 objective입니다.

### [[concepts/generative-models/conditional-generation|Conditional generation]]

LiFT는 condition $c$를 static concatenation이 아니라 dynamic routing signal로 확장한 사례로 읽을 수 있습니다.

### [[concepts/generative-models/molecular-generation|Molecular generation]]

Validity/diversity/novelty/task utility의 metric boundary가 특히 중요합니다.

### [[molecular-modeling/structure-based/index|Structure-based modeling]]

Pocket-conditioned generation에서 pose, affinity, screening utility가 서로 다른 claim임을 유지해야 합니다.

### [[papers/generative-models/molexar|Molexar]]

둘 다 multimodal chemical conditioning을 다루지만, Molexar는 unified autoregressive representation 쪽이고 LiFT는 semantic prior와 3D geometric flow의 **interface**에 더 초점이 있습니다.

## 23. Final verdict

**Verdict: Must Read for 3D molecular generation / cross-modal conditioning.**

이 논문의 장점은 `LLM을 썼다`가 아닙니다. 더 일반적으로 재사용할 수 있는 설계 원리는 다음 두 가지입니다.

1. **semantic model과 geometric model의 책임을 분리한다.**
2. **semantic condition의 영향력을 현재 geometric state에 따라 route한다.**

Evidence는 흥미롭고 공개 artifact도 좋은 편이지만, main benchmark가 CrossDocked2020 중심이므로 broad OOD/generalization claim은 아직 열어두는 것이 맞습니다. 특히 chemistry metric 개선과 PoseBusters trade-off를 같이 읽어야 합니다.

## 6개월 뒤 기억해야 할 세 가지

1. **LiFT는 LLM이 3D molecule을 직접 생성하는 모델이 아니다.** LLM-derived condition은 semantic prior이고 final geometry는 Flow Matching generator가 만든다.
2. **SCDR가 핵심 architecture idea다.** 같은 condition을 모든 ODE step에 고정 주입하지 않고 current 3D state에 따라 semantic guidance를 조절한다.
3. **결과는 trade-off다.** medicinal-chemistry metrics와 filter compliance는 크게 좋아지지만 모든 structural-validity metric을 동시에 지배하지는 않으며, strong pocket/scaffold OOD 검증은 여전히 필요하다.

## Sources

- [arXiv:2608.31009](https://arxiv.org/abs/2608.31009)
- [Official implementation: kasurl/LiFT](https://github.com/kasurl/LiFT)
- [OpenReview](https://openreview.net/forum?id=ds3gmBwsR2)
- [DrugFlow](https://github.com/LPDI-EPFL/DrugFlow)
