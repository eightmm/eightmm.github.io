---
title: NEAT-POCKET — Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer
aliases:
  - papers/neat-pocket
  - papers/sbdd/neat-pocket
tags:
  - papers
  - sbdd
  - structure-based-modeling
  - molecular-generation
  - autoregressive
  - flow-matching
  - set-transformer
  - protein-ligand
  - fragment-completion
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2609.05097
---

# NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer

> **한 줄 요약:** NEAT-POCKET의 핵심은 pocket-conditioned 3D generation을 수백 번의 전역 denoising으로 풀지 않고, **현재까지 만들어진 unordered atom set에서 다음 원자의 type과 3D 위치를 한 번씩 결정하는 autoregressive outer loop**로 바꾸면서, pocket은 fine→coarse→fine transformer와 cross-attention/AdaLN으로 조건화하고 fragment prefix는 초기 state로 그대로 넣어 lead-optimization형 completion을 자연스럽게 지원한다는 것입니다.

## 왜 이 논문을 저장하는가

Pocket-conditioned molecular generation은 최근 diffusion과 flow 계열이 지배적입니다. 이들은 강한 geometric inductive bias와 parallel state refinement라는 장점이 있지만, 한 molecule을 만들기 위해 같은 비싼 network를 여러 noise/time step에서 반복 평가합니다.

추상적으로 iterative generator의 비용을

$$
C_{\mathrm{iter}}
\approx
N_{\mathrm{step}}\,C_{\mathrm{global}}
$$

라고 두면, $N_{\mathrm{step}}$이 수십~수백이고 매 step마다 ligand와 pocket 전체를 다시 처리할 때 sampling latency가 커집니다.

NEAT-POCKET은 질문을 바꿉니다.

> **분자를 한 번에 계속 정제하지 말고, pocket을 보면서 원자를 하나씩 추가하면 어떨까?**

이때 단순한 canonical-order autoregressive model로 돌아가지 않는 것이 중요합니다. Base NEAT의 핵심은 현재 molecule을 **ordered sequence가 아니라 set**으로 다루고, 임의 source set에서 그 1-hop neighborhood를 target으로 예측하는 neighborhood-guided training입니다. 따라서 generation order 자체가 molecular identity의 일부가 되지 않도록 설계합니다.

그리고 한 가지 더 중요한 nuance가 있습니다.

> **NEAT-POCKET은 “flow를 버린 autoregressive model”이 아닙니다.**

Outer generation loop는 atom-by-atom autoregressive이지만, **새 원자의 3D coordinate를 놓는 local head는 continuous Flow Matching**입니다. 즉 이 논문의 진짜 비교 축은

$$
\text{global iterative flow/diffusion}
\quad\text{vs}\quad
\text{autoregressive set construction + local flow head}
$$

입니다.

이 분해는 SBDD generative model을 설계할 때 매우 재사용 가능한 architecture choice입니다.

---

## Metadata

| Field | Value |
| --- | --- |
| Paper | NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer |
| Authors | Roxane Axel Jacob, Daniel Rose, Thierry Langer, Johannes Kirchmair |
| arXiv | [2609.05097](https://arxiv.org/abs/2609.05097) |
| Version | v1 |
| Submitted | 2026-09-04 |
| Base model | NEAT |
| Generation family | Set-based autoregressive outer loop + Flow Matching coordinate head |
| Main datasets | CrossDocked, SPINDR |
| Official implementation | [molinfo-vienna/NEAT-POCKET](https://github.com/molinfo-vienna/NEAT-POCKET) |
| Reviewed code snapshot | [`323f262`](https://github.com/molinfo-vienna/NEAT-POCKET/tree/323f262b17e1ee03f18e8dbaffe31088c6df2993) |
| Released weights | [Figshare 10.6084/m9.figshare.33426877](https://doi.org/10.6084/m9.figshare.33426877) |
| Code license | MIT |
| Main practical capability | de novo pocket generation + fragment/prefix completion |

> **Claim boundary:** 아래 benchmark 수치와 qualitative conclusions는 논문 저자들이 보고한 결과입니다. PoseBusters validity, protein–ligand clash count, strain energy, docking score, molecular size, sampling time, 그리고 실제 binding/activity는 서로 다른 evidence layer이며 하나의 “drug quality” 점수로 합치지 않습니다.

---

## Figure guide — 먼저 이 두 장을 보자

### Official overview — architecture와 generation loop

![NEAT-POCKET overview](https://raw.githubusercontent.com/molinfo-vienna/NEAT-POCKET/323f262b17e1ee03f18e8dbaffe31088c6df2993/images/overview.png)

*Source: official NEAT-POCKET repository, `images/overview.png`, commit [`323f262`](https://github.com/molinfo-vienna/NEAT-POCKET/blob/323f262b17e1ee03f18e8dbaffe31088c6df2993/images/overview.png), MIT-licensed repository. 이 그림은 저자 제공 architecture visualization이며 독립 검증 결과가 아닙니다.*

**볼 것:** ligand를 한꺼번에 denoise하는 것이 아니라 현재 molecular set을 context로 다음 atom을 추가하고, pocket representation이 ligand stream에 conditioning으로 들어가는 흐름입니다. 이 그림을 읽을 때 가장 중요한 것은 “autoregressive”와 “3D flow”가 상호 배타적이지 않다는 점입니다.

### Official prefix artifact — fragment completion의 입력 contract

![NEAT-POCKET released prefix examples](https://raw.githubusercontent.com/molinfo-vienna/NEAT-POCKET/323f262b17e1ee03f18e8dbaffe31088c6df2993/prefixes/prefixes.png)

*Source: official NEAT-POCKET repository, `prefixes/prefixes.png` and paired `prefixes/prefixes.sdf`, commit [`323f262`](https://github.com/molinfo-vienna/NEAT-POCKET/tree/323f262b17e1ee03f18e8dbaffe31088c6df2993/prefixes). 이 이미지는 released fragment/prefix artifact를 보여주는 공식 repository visualization입니다.*

**볼 것:** prefix-conditioned generation은 별도의 mask/inpainting trajectory를 정의하는 대신, 이미 주어진 fragment를 현재 source set으로 넣고 그 뒤 atom을 계속 추가하는 interface입니다. 이 때문에 prefix identity와 coordinates를 “다시 생성해서 맞추는” 문제가 아니라 **초기 condition으로 보존하는 문제**가 됩니다.

---

## 1. Problem: 3D molecular generation의 비용은 어디서 생기는가

Pocket-conditioned generation의 목표는 보통 다음과 같이 적을 수 있습니다.

$$
x_{\mathrm{lig}}
\sim
p_\theta(x_{\mathrm{lig}}\mid P),
$$

where

- $P$: target protein pocket,
- $x_{\mathrm{lig}}$: atom types, coordinates, connectivity를 포함하는 generated ligand입니다.

Diffusion/flow 계열에서는 보통 noisy full-ligand state $x_t$를 여러 번 업데이트합니다.

$$
x_{t+\Delta t}
=
x_t
+
v_\theta(x_t,P,t)\Delta t.
$$

각 step에서 전체 ligand state와 pocket interaction을 다시 계산하므로 inference cost는 number of function evaluations와 강하게 연결됩니다.

반대로 autoregressive factorization은

$$
p(x_{\mathrm{lig}}\mid P)
=
\prod_{k=1}^{N}
p(a_k,r_k \mid S_{k-1},P)
\cdot
p(\mathrm{STOP}\mid S_N,P),
$$

처럼 생각할 수 있습니다.

여기서

- $S_{k-1}$: 지금까지 생성된 atom **set**,
- $a_k$: 다음 atom type,
- $r_k$: 다음 atom coordinate입니다.

단순히 보면 atom 수만큼 순차 step이 필요하기 때문에 autoregression도 “공짜”는 아닙니다. 하지만 한 step에서 다루는 task가 **전체 molecule 재정제**가 아니라 **새 atom 하나의 local decision**으로 바뀝니다.

따라서 비교해야 할 것은 단순 step count가 아니라

$$
\text{wall time},\quad
\text{network calls},\quad
\text{quality per generated molecule}
$$

입니다.

NEAT-POCKET의 가장 강한 empirical claim은 바로 이 축입니다.

---

## 2. Base NEAT의 핵심: sequence가 아니라 set을 autoregress한다

일반적인 autoregressive molecular model은 atom ordering을 하나 정합니다.

```text
atom 1 → atom 2 → atom 3 → ...
```

그러면 동일한 molecule도 atom indexing이나 traversal rule에 따라 다른 sequence가 될 수 있습니다. 이것은 molecule이 본질적으로 permutation-invariant object라는 사실과 충돌합니다.

NEAT의 training contract는 다릅니다.

현재 molecule의 atom subset을 source set $S$라고 하고, target을 source set의 one-hop molecular neighborhood로 둡니다.

$$
S
\longrightarrow
\mathcal N_1(S).
$$

Official implementation에서도 forward pass가 molecular data를 source atom set과 target atom set으로 나누고, target set을 source의 1-hop neighborhood로 정의합니다.

이렇게 하면 model이 학습하는 질문은

> “다음 index가 무엇인가?”

가 아니라

> “현재 존재하는 이 atom set의 chemically local frontier에 무엇이 추가되어야 하는가?”

가 됩니다.

이 차이가 `Neighborhood-Guided Set Transformer`라는 이름의 핵심입니다.

### 왜 이것이 중요한가

Permutation invariance는 단순 미학적 성질이 아닙니다.

- 특정 canonical order에 과도하게 의존하지 않습니다.
- arbitrary molecular fragment를 source set으로 사용할 수 있습니다.
- fragment completion을 별도 architecture로 만들 필요가 줄어듭니다.
- source set의 크기가 0부터 full molecule까지 바뀌는 training task가 자연스럽습니다.

즉 **generation order를 representation contract에서 제거**합니다.

---

## 3. 한 step에서 실제로 무엇을 예측하는가

한 autoregressive step은 크게 두 문제입니다.

### 3.1 Atom type

현재 set representation $h_S$에서 다음 atom class를 예측합니다.

$$
p_\theta(a_{\mathrm{next}}\mid S,P)
=
\operatorname{softmax}(W h_S).
$$

STOP token도 같은 decision family 안에 들어갑니다.

### 3.2 Coordinate

새 atom의 coordinate는 단순 regression으로 찍지 않습니다.

Official implementation은 `SimpleMLPAdaLN` 기반의 **Flow Matching head**를 사용합니다. 즉 random/noisy position에서 target coordinate로 가는 local velocity field를 학습합니다.

추상적으로

$$
\frac{d r_t}{dt}
=
v_\theta(r_t,t\mid h_S,a_{\mathrm{next}},P).
$$

중요한 점은 $r_t$가 **새로 추가될 target atom의 위치**라는 것입니다.

그래서 NEAT-POCKET의 계산 구조는 다음처럼 볼 수 있습니다.

```text
current atom set S_k
    ↓ set transformer + pocket conditioning
next atom type
    ↓
local coordinate flow
    ↓
add one atom to set
    ↓
S_{k+1}
```

전역 3D state에 대한 수백-step trajectory 대신 **짧은 local coordinate-generation problem을 atom addition마다 푸는 구조**입니다.

---

## 4. Pocket encoder: fine → coarse → fine

Pocket atom을 모두 flat token으로 넣으면 local atomic detail은 잘 보이지만 long-range context 비용이 커집니다. 반대로 residue token만 쓰면 atom-level steric detail을 잃을 수 있습니다.

NEAT-POCKET은 pocket을 세 단계로 처리합니다.

### Stage A — atom-level pocket transformer

Pocket atom type과 Cartesian coordinates를 embedding하고 atom-level self-attention을 수행합니다.

$$
\{(z_i,r_i)\}_{i=1}^{N_P}
\rightarrow
\{h_i^{\mathrm{atom}}\}.
$$

### Stage B — atom → residue pooling + residue transformer

같은 residue에 속한 atom representation을 pool합니다. Reviewed official training config의 default는 **sum pooling**입니다.

$$
h_j^{\mathrm{res}}
=
\sum_{i\in \mathrm{res}(j)}
h_i^{\mathrm{atom}}.
$$

그 뒤 residue-level transformer가 coarse context를 교환합니다.

### Stage C — residue information을 다시 atom resolution으로

Residue-level context를 다시 atom stream과 결합한 뒤 두 번째 atom-level pocket transformer를 적용합니다.

따라서 pocket representation은

$$
\boxed{
\text{atom}
\rightarrow
\text{residue}
\rightarrow
\text{atom}
}
$$

입니다.

이 구조는 pocket에서 필요한 두 종류의 정보가 다르다는 가정을 담습니다.

- atom level: clash, local chemistry, precise geometry
- residue level: broader pocket context, interaction environment

이 fine–coarse–fine contract는 pocket model 자체로도 재사용 가치가 있습니다.

---

## 5. Pocket과 ligand는 어떻게 만나는가

Pocket representation을 얻은 다음 ligand stream에 두 경로로 조건을 제공합니다.

### 5.1 Cross-attention

Ligand source-set token이 pocket atom representation을 직접 참조할 수 있습니다.

$$
H_L'
=
\operatorname{CrossAttn}(H_L,H_P).
$$

이를 통해 “현재 fragment/partial molecule의 이 부분에서 pocket의 어떤 부분을 봐야 하는가?”를 local하게 결정할 수 있습니다.

### 5.2 Global adaptive normalization

Pocket representation을 graph-level로 pool해 global condition $c_P$를 만들고, ligand transformer / coordinate head의 normalization을 조절합니다.

개념적으로

$$
\operatorname{AdaLN}(h;c_P)
=
\gamma(c_P)\odot \operatorname{LN}(h)
+
\beta(c_P).
$$

Official training config에서는 `global_cond_proj: true`를 사용합니다.

### 5.3 Zero initialization

새 conditioning branch를 pretrained unconditional NEAT에 붙일 때 cross-attention projection과 adaptive-conditioning projection의 마지막 layer를 **zero-init**합니다.

초기에는

$$
f_{\mathrm{conditional}}
\approx
f_{\mathrm{pretrained}},
$$

이 되도록 만들어 pocket condition이 학습 시작부터 pretrained molecular prior를 강하게 깨뜨리지 않게 합니다.

이것은 conditional fine-tuning에서 매우 일반적으로 재사용 가능한 design pattern입니다.

---

## 6. Pretraining → pocket conditioning

Released training config는 pretrained unconditional NEAT를 사용합니다.

공식 config의 주요 값은 다음과 같습니다.

| Parameter | Released SPINDR config |
| --- | ---: |
| Transformer width | 768 |
| Ligand transformer layers | 12 |
| Attention heads | 12 |
| Flow-head width | 1536 |
| Flow-head residual blocks | 6 |
| Residue-level pocket layers | 4 |
| Atom-level pocket layers | 1 before + 1 after residue stage |
| Batch size | 128 |
| Learning rate | $5\times10^{-5}$ |
| Max epochs | 5000 |
| CFG dropout | 0.2 |
| Clash penalty weight | 4.0 |
| Freeze pretrained model | yes initially |
| Unfreeze epoch | 150 |
| Flow noise std | 2.5 |
| Time-step resampling | 4 |

Official code는 matching pretrained parameters를 conditional model로 옮긴 뒤, 설정에 따라 pretrained branch를 먼저 freeze하고 새 pocket-conditioning layers를 학습할 수 있게 합니다.

이 training schedule의 의도는 명확합니다.

$$
\text{general molecular prior}
\rightarrow
\text{pocket interface adaptation}
\rightarrow
\text{joint fine-tuning}.
$$

---

## 7. Clash penalty는 무엇을 학습시키는가

Pocket-conditioned generator가 좋은 molecular distribution을 배웠다고 해서 protein volume을 자동으로 피하는 것은 아닙니다.

NEAT-POCKET은 explicit protein–ligand clash penalty를 추가합니다.

일반적으로 atom pair distance $d_{ij}$가 허용 거리보다 작을 때 penalty를 주는 형태로 생각할 수 있습니다.

$$
L_{\mathrm{clash}}
=
\lambda
\sum_{i\in L}
\sum_{j\in P}
\phi(d_{ij}),
$$

where $\phi$는 너무 가까운 pair에 대해 증가합니다.

전체 objective는 개념적으로

$$
L
=
L_{\mathrm{atom}}
+
L_{\mathrm{flow}}
+
L_{\mathrm{clash}}.
$$

Official implementation의 forward path도 atom-type CE, flow-matching loss, optional clash penalty를 합칩니다.

이 항의 역할은 docking affinity를 최적화하는 것이 아닙니다.

> **Pocket condition을 “protein과 충돌하지 않는 geometry”라는 최소 geometric constraint로 연결하는 것**입니다.

따라서 clash 감소를 binding 개선으로 해석하면 안 됩니다.

---

## 8. Classifier-Free Guidance: condition을 강하게 하면 항상 좋아지는가?

아닙니다. 이 논문의 중요한 negative result 중 하나입니다.

Training에서는 일정 확률로 pocket condition을 drop하여 conditional/unconditional branch를 함께 학습하고, sampling에서는 classifier-free guidance로 pocket conditioning strength를 조절합니다.

추상적으로

$$
v_{\mathrm{cfg}}
=
v_{\mathrm{uncond}}
+
w
\left(
v_{\mathrm{cond}}-v_{\mathrm{uncond}}
\right).
$$

$w$가 커지면 pocket preference가 강해질 수 있지만, molecular prior를 과도하게 밀어낼 수도 있습니다.

CrossDocked에서 저자 보고 pattern은 다음과 같습니다.

| CFG factor | PB valid ↑ | clashes ↓ | strain ↓ | Vina ↓ | MW |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 71.0 | 8.3 | 155 | -4.4 | 273 |
| 0.5 | 69.2 | 6.6 | 265 | -5.1 | 289 |
| 1.0 | 64.1 | 5.8 | 382 | -5.3 | 300 |
| 1.5 | 58.4 | 5.1 | 511 | -5.5 | 308 |
| 2.0 | 53.5 | 4.7 | 544 | -5.6 | 316 |
| 2.5 | 47.9 | 4.5 | 783 | -5.6 | 323 |

해석은 단순합니다.

- stronger conditioning → clash 감소
- raw Vina score는 좋아짐
- molecule size 증가
- PoseBusters validity 하락
- strain 급증

즉

$$
\text{more pocket guidance}
\neq
\text{uniformly better molecule}.
$$

특히 Vina는 ligand size와 강하게 연결되므로 $w$가 커질수록 Vina가 좋아지는 것을 순수한 pocket complementarity 개선으로 읽으면 안 됩니다.

이 paper가 보여주는 것은 **controllability–physical-plausibility frontier**입니다.

---

## 9. Benchmark contract: CrossDocked와 SPINDR를 같은 의미로 읽으면 안 된다

### CrossDocked

CrossDocked는 historical comparability가 좋지만 rigid cross-docking 때문에 reference complex 자체에 비현실적 geometry가 포함될 수 있습니다.

논문이 평가한 reference ligand도 100% PoseBusters-valid하지 않습니다. 따라서 model-generated structure의 absolute quality를 판단할 때 benchmark ceiling 자체가 깨끗하지 않습니다.

또 원래 CrossDocked ligand에는 hydrogen annotation이 거의 없어, NEAT-POCKET은 explicit-hydrogen model과 consistent evaluation을 위해 preprocessing에서 hydrogens를 복원합니다.

### SPINDR

SPINDR는 PLINDER 기반으로 더 엄격한 filtering과 protein preparation을 적용한 high-quality complex set입니다.

이 benchmark를 포함한 것은 이 논문의 장점입니다. 단순히 CrossDocked leaderboard 하나에서만 model을 평가하지 않습니다.

하지만 SPINDR도 retrospective computational benchmark입니다.

따라서 다음 claim은 아직 별개입니다.

$$
\text{SPINDR 3D validity}
\not\Rightarrow
\text{prospective binding}
\not\Rightarrow
\text{synthesizable active compound}.
$$

---

## 10. Main result: speed는 강하고, quality는 trade-off다

### CrossDocked

저자 보고 핵심 비교는 다음과 같습니다.

| Model | PB valid ↑ | clashes ↓ | strain ↓ | Vina ↓ | Vina-min ↓ | MW | 100 mol runtime ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pocket2Mol | 71.1 | 6.9 | 65 | -5.4 | -6.9 | 223 | 845 s |
| TargetDiff | 46.4 | 10.8 | 1003 | -6.5 | -7.5 | 324 | 722 s |
| DiffSBDD | 49.6 | 14.3 | 891 | -3.6 | -6.3 | 296 | 88 s |
| DrugFlow | 61.7 | 9.2 | 255 | -5.8 | -6.9 | 309 | 112 s |
| **NEAT-POCKET** | **69.2** | **6.6** | 265 | -5.1 | -6.6 | 289 | **4 s** |

이 표에서 정직한 결론은 “NEAT-POCKET이 best generator”가 아닙니다.

- PB validity는 강합니다.
- clash count는 낮습니다.
- sampling speed는 매우 강합니다.
- strain은 Pocket2Mol보다 나쁩니다.
- docking score는 TargetDiff/DrugFlow보다 약합니다.
- generated molecular size가 모델마다 다릅니다.

즉 가장 강한 claim은

> **competitive geometric quality를 유지하면서 sampling latency를 크게 줄였다**

입니다.

### SPINDR

SPINDR에서는 비교 가능한 FLOWR와 다음 trade-off가 보고됩니다.

| Model | PB valid ↑ | clashes ↓ | strain ↓ | Vina ↓ | Vina-min ↓ | MW | runtime ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOWR | 75.8 | 4.6 | 35 | -7.0 | -7.7 | 395 | 36 s |
| **NEAT-POCKET** | **80.3** | **3.7** | 62 | -5.5 | -6.8 | 307 | **4 s** |

여기서도 same story입니다.

- validity / clashes / speed → NEAT-POCKET 우위
- strain / docking → FLOWR 우위
- FLOWR molecule이 훨씬 큼

따라서 docking score만으로 architecture ranking을 하면 size confound가 큽니다.

---

## 11. Molecular-size confound를 반드시 따로 봐야 한다

Vina-like docking score는 대체로 더 많은 contact를 만들 수 있는 큰 molecule에 유리해질 수 있습니다.

그래서

$$
E_{\mathrm{Vina}}
\sim
f(\text{fit},\text{contacts},\text{size},\ldots)
$$

이고 단순히 score가 더 낮다고 해서 normalized binding quality가 높다는 뜻은 아닙니다.

NEAT-POCKET은 CrossDocked에서 평균 MW 약 289 Da, SPINDR에서 약 307 Da의 비교적 작은 molecule을 생성합니다.

이는 두 방향으로 작용합니다.

### 장점

- early hit처럼 optimization headroom이 남을 수 있음
- steric clash가 줄기 쉬움
- fragment growth에 적합할 수 있음

### 단점 / confound

- raw Vina가 불리할 수 있음
- “good pocket filling”과 “small ligand”가 섞일 수 있음

따라서 fair evaluation에는 최소한 다음을 같이 봐야 합니다.

- heavy-atom count / MW
- ligand efficiency-like normalized score
- pocket occupancy
- clash count
- strain
- pose validity

---

## 12. Fragment completion이 왜 architecture-level 장점인가

이 논문에서 가장 실용적으로 흥미로운 부분은 de novo generation보다 fragment completion일 수 있습니다.

Diffusion inpainting에서는 보통 prefix atom을 trajectory 중 고정하거나, 매 step reference position으로 re-project하거나, mask-specific dynamics를 설계해야 합니다.

NEAT-POCKET에서는 fragment $F$를 그냥 initial source set으로 둡니다.

$$
S_0 = F.
$$

그 뒤

$$
S_{k+1}
=
S_k \cup \{a_{k+1},r_{k+1}\}.
$$

즉 기존 fragment를 다시 생성하지 않습니다.

따라서 prefix preservation은 soft objective가 아니라 construction rule에 가깝습니다.

> **이미 존재하는 원자를 건드리지 않고 frontier만 성장시킨다.**

Lead optimization / scaffold elaboration과 interface가 잘 맞는 이유입니다.

---

## 13. Fragment size가 커질수록 무슨 일이 생기는가

저자들의 prefix experiments는 intuitive한 trade-off를 보여줍니다.

CrossDocked에서 large prefix를 주면 de novo보다:

- PB validity: 약 69.2% → 81.8%
- strain: 약 265 → 78 kcal/mol
- Vina: 약 -5.1 → -6.1 kcal/mol

SPINDR에서는:

- PB validity: 약 80.3% → 88.8%
- strain: 약 62 → 25 kcal/mol
- Vina: 약 -5.5 → -7.0 kcal/mol

즉 더 많은 chemically valid structure를 고정해 주면 completion problem이 쉬워집니다.

하지만 diversity cost가 생깁니다.

- CrossDocked uniqueness: 약 92.8% → 42.4%
- SPINDR uniqueness: 약 97.6% → 52.2%

이것은 실패라기보다 조건부 generation의 본질적인 trade-off입니다.

$$
\text{more fixed structure}
\Rightarrow
\text{less search space}
\Rightarrow
\text{higher local feasibility, lower diversity}.
$$

따라서 fragment completion benchmark는 validity 하나만 보면 안 됩니다.

---

## 14. Prefix preservation은 별도 metric이어야 한다

Fragment-conditioned generation에서 아주 중요한 평가 질문은:

> 결과 molecule이 “그 fragment와 비슷한가?”가 아니라 **정확히 그 fragment를 유지했는가?**

입니다.

NEAT-POCKET은 construction상 prefix coordinates와 atom identity를 유지합니다.

반면 다른 generative method가 fragment conditioning을 지원하더라도 trajectory 중 fragment가 변형될 수 있습니다.

따라서 fragment benchmark에는 다음 metric이 필요합니다.

- exact atom/bond identity retention
- prefix RMSD
- attachment-point correctness
- completed molecule validity
- pocket clash
- strain
- docking
- uniqueness/diversity
- synthetic plausibility

특히 prefix retention filter 없이 전체 결과만 평균내면 “fragment-conditioned”라는 claim 자체가 약해질 수 있습니다.

---

## 15. Bond generation은 coordinate generation과 분리되어 있다

NEAT-POCKET은 atom type과 coordinates를 생성한 뒤 bond assignment를 별도 predictor/reconstruction stage에서 처리합니다.

이 design은 장점과 비용이 있습니다.

### 장점

3D coordinate generation problem을 bond-order combinatorics와 분리할 수 있습니다.

### 위험

좋은 point cloud가 항상 chemically correct graph로 변환되지는 않습니다.

즉 최종 validity는

$$
\text{3D placement quality}
+
\text{bond reconstruction quality}
$$

의 합성 결과입니다.

따라서 failure analysis에서도 다음을 분리해야 합니다.

1. atom type failure
2. coordinate/steric failure
3. bond assignment failure
4. sanitization failure

“generation invalid” 하나로 합치면 architecture bottleneck을 찾기 어렵습니다.

---

## 16. 이 모델은 equivariant network가 아닌데 rotation 문제는 어떻게 다루는가

NEAT family는 전형적인 irreps-based SE(3) network를 backbone으로 사용하지 않습니다. Cartesian coordinate를 Fourier feature로 embedding한 transformer 계열입니다.

이 경우 중요한 질문은:

> coordinate frame dependence를 어떻게 막는가?

입니다.

Official code에는 random rotation augmentation이 포함되어 있습니다.

즉 strict architectural equivariance 대신

$$
\text{augmentation}
+
\text{set symmetry}
+
\text{coordinate featurization}
$$

으로 rotation robustness를 학습하는 방향입니다.

이것은 매우 중요한 비교 축입니다.

### Strict equivariance의 장점

$$
f(Rx+t)=Rf(x)+t
$$

같은 transform law가 architecture에 내장됩니다.

### Learned/augmented geometric behavior의 장점

- simpler/faster kernels 가능
- higher-order irreps overhead 없음
- 일반 transformer stack 재사용 가능

따라서 NEAT-POCKET의 속도 gain을 해석할 때 단순 “autoregression이 diffusion보다 빠르다”라고만 하면 불완전합니다.

**Backbone symmetry implementation의 비용 차이도 포함**될 수 있습니다.

---

## 17. 무엇이 실제 novelty인가

구성 요소를 따로 보면 완전히 처음인 것은 많지 않습니다.

### Autoregressive molecule generation

기존에 존재합니다.

### Set Transformer

기존 architecture family입니다.

### Pocket cross-attention

다른 conditional molecular model에서도 볼 수 있습니다.

### Flow Matching

coordinate generation objective 자체도 기존입니다.

### Clash penalty / CFG

개별 기술은 알려져 있습니다.

### 진짜 기여

재사용 가능한 contribution은 이 조합에 있습니다.

$$
\boxed{
\text{permutation-invariant set autoregression}
+
\text{local coordinate flow}
+
\text{fine/coarse pocket conditioning}
+
\text{prefix-as-state interface}
}
$$

특히 **fragment completion을 별도 inpainting algorithm이 아니라 autoregressive state initialization으로 바꾼 것**이 architecture–task alignment 측면에서 강합니다.

---

## 18. 이 논문이 잘 보여주는 것

### 18.1 Global iterative refinement가 유일한 3D generation route는 아니다

Competitive validity를 유지하면서 훨씬 낮은 sampling latency가 가능하다는 evidence를 제공합니다.

### 18.2 Geometric metric은 하나가 아니다

PoseBusters, clash, strain, docking이 서로 다른 방향으로 움직입니다.

### 18.3 Dataset quality가 model ranking을 바꾼다

CrossDocked와 SPINDR를 함께 보면 “좋은 generator”의 의미가 benchmark preprocessing에 강하게 의존한다는 점이 드러납니다.

### 18.4 Prefix completion은 autoregressive representation과 잘 맞는다

Fixed fragment를 trajectory constraint로 다루지 않고 state로 보존할 수 있습니다.

### 18.5 Conditioning strength에는 frontier가 있다

Strong CFG가 clash와 raw docking을 개선하면서 동시에 strain/validity를 망가뜨릴 수 있습니다.

---

## 19. 이 논문이 아직 증명하지 못한 것

### 19.1 Prospective biological activity

모든 주 evidence는 computational generation benchmark입니다.

$$
\text{good generated pose metrics}
\not\Rightarrow
\text{active compound}.
$$

### 19.2 Broad protein-family OOD

CrossDocked/SPINDR split에서 잘 작동하는 것과 unseen protein families에 일반화하는 것은 다릅니다.

필요한 것은 explicit:

- protein sequence/structure family split
- ligand scaffold split
- joint protein + scaffold OOD
- temporal structure split

입니다.

### 19.3 Apo pocket discovery

Released `generation_from_pdb.py` workflow는 현재 reference ligand가 pocket region을 정의하는 데 필요합니다. README는 ligand가 generation condition으로 직접 사용되지는 않는다고 명시하지만, **known ligand location으로 pocket을 정한다는 사실 자체는 deployment information**입니다.

따라서 이것은 global de novo pocket discovery model이 아닙니다.

### 19.4 Protein flexibility

Pocket은 기본적으로 static structure입니다.

- induced fit
- alternative side-chain conformations
- water network
- ensemble uncertainty

를 해결하지 않습니다.

### 19.5 Autoregressive error accumulation

앞 단계의 잘못된 atom addition이 뒤 단계 context가 됩니다.

Molecule이 커질수록 error propagation이 누적될 수 있습니다.

---

## 20. 가장 중요한 compute-matched ablation

이 architecture를 판단하려면 다음 비교가 가장 중요합니다.

| Arm | Generator | Pocket conditioning | Prefix support |
| --- | --- | --- | --- |
| A | iterative global flow | same pocket encoder | inpainting |
| B | autoregressive set + coordinate regression | same | native |
| C | autoregressive set + local flow | same | native |
| D | C + clash penalty | same | native |
| E | D + CFG | same | native |

모든 arm에서:

- training complexes
- pocket cutoff
- parameter budget
- generated molecule count
- GPU
- final bond reconstruction
- evaluation scripts

를 가능한 한 맞춰야 합니다.

### Primary outputs

$$
\text{quality per wall-clock / GPU-second}
$$

를 중심으로:

- PB validity
- protein–ligand clashes
- strain
- normalized docking
- MW / heavy atom count
- diversity / novelty
- exact prefix retention
- model calls / NFE
- peak memory
- throughput

를 같이 봐야 합니다.

---

## 21. Autoregressive vs diffusion/flow를 더 공정하게 비교하려면

“4초 vs 88초”는 강한 숫자지만 architecture science로는 다음 요소를 분리해야 합니다.

### 21.1 Backbone cost

Equivariant GNN/Transformer와 ordinary set transformer의 FLOPs 차이.

### 21.2 Number of updates

Global denoising NFE와 atom additions 수.

### 21.3 Batch parallelism

Autoregressive generation이 molecule 내부에서는 sequential이어도 여러 molecules를 batch로 병렬화할 수 있습니다. Released README는 RTX 4090에서 큰 generation batch도 가능하다고 설명합니다.

### 21.4 Bond post-processing

Generation runtime에 bond reconstruction / sanitization / evaluation이 어디까지 포함되는지 동일하게 맞춰야 합니다.

### 21.5 Molecule size

더 작은 molecule은 generation step 수 자체가 적습니다.

따라서 throughput은

$$
\text{molecules/sec}
$$

뿐 아니라

$$
\text{generated heavy atoms/sec}
$$

로도 보는 것이 좋습니다.

---

## 22. Generalization test는 이렇게 강화할 수 있다

### Protein-family OOD

Protein sequence identity나 pocket structure similarity로 cluster한 뒤 cluster-disjoint split.

### Ligand scaffold OOD

Bemis–Murcko scaffold가 train/test를 넘지 않도록 split.

### Joint OOD

$$
\text{new pocket family}
+
\text{new ligand scaffold}.
$$

### Size-stratified evaluation

Ligand heavy-atom count별로 performance를 나눠 autoregressive error accumulation을 확인합니다.

### Prefix difficulty axis

Prefix size뿐 아니라 attachment-point ambiguity와 3D exposure를 나눕니다.

예:

- buried prefix
- edge-exposed prefix
- multiple growth vectors
- constrained macrocyclic context

### Pocket uncertainty

Same target의 multiple experimental conformations 또는 predicted-structure ensemble을 사용합니다.

---

## 23. Lead optimization 관점에서 가장 흥미로운 실험

Fragment completion의 진짜 utility는 단순 random BRICS prefix보다 medicinal-chemistry setting에서 확인해야 합니다.

예를 들면:

```text
known fragment / lead core
    ↓
fixed interaction motif
    ↓
NEAT-POCKET growth
    ↓
pose/strain/clash filters
    ↓
novel substituent diversity
    ↓
rescoring / synthesis feasibility
```

여기서 중요한 metric은:

- core RMSD = 0에 가까운가
- key interaction 유지
- growth vector 다양성
- R-group novelty
- synthetic accessibility
- docking improvement normalized by added heavy atoms
- strain increase
- scaffold/analog diversity

입니다.

특히 “더 큰 molecule을 붙이면 docking score가 좋아진다”는 trivial solution을 막기 위해

$$
\Delta \text{score}/\Delta N_{\mathrm{heavy}}
$$

같은 normalized view가 필요합니다.

---

## 24. Reproducibility path

Official repository는 재현성이 좋은 편입니다.

### 공개된 것

- training code
- generation code
- fragment-generation utility
- evaluation script
- bond predictor training
- CrossDocked/SPINDR dataset loaders
- pretrained / conditional weight download path
- released YAML configs
- PoseBusters and SBDD metric utilities
- official architecture image
- example protein/ligand/fragment input

### 권장 재현 순서

1. Released weights를 받습니다.
2. Official example `3R8G`로 `generation_from_pdb.py`가 돌아가는지 확인합니다.
3. 동일 RTX-class GPU에서 100-molecule runtime을 측정합니다.
4. SPINDR released/default config로 PB validity / clashes를 재현합니다.
5. CFG factor sweep를 재현합니다.
6. BRICS prefix experiment를 실행합니다.
7. Prefix identity/RMSD를 독립적으로 다시 검증합니다.
8. 그 다음 OOD split을 추가합니다.

### 주의

README 기준으로 현재 arbitrary PDB generation은 reference ligand가 pocket extraction에 필요합니다. 이 조건을 “protein structure만으로 generation”이라고 축약해서는 안 됩니다.

---

## 25. Failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| 4 s runtime만 보고 superior generator라 결론 | quality/size/backbone cost가 다름 |
| raw Vina로 모델 ranking | molecular-size confound |
| low clash = high affinity로 해석 | steric feasibility와 binding thermodynamics는 다름 |
| CrossDocked만으로 deployment 주장 | rigid cross-docking artifact |
| SPINDR success를 prospective activity로 확장 | retrospective structural benchmark |
| prefix completion에서 retention metric 생략 | fragment가 실제로 보존됐는지 모름 |
| large prefix의 validity 향상을 모델 capability로만 해석 | search space가 줄어든 효과가 큼 |
| autoregressive molecule size별 실패율을 합쳐 보고 | error accumulation이 숨겨질 수 있음 |
| known ligand로 정의한 pocket을 apo setting으로 표현 | deployment-unavailable location leakage 가능 |
| static pocket을 induced-fit 해결로 해석 | protein flexibility는 별도 문제 |

---

## 26. 다른 개념과의 연결

### [[concepts/generative-models/autoregressive-generation|Autoregressive generation]]

NEAT-POCKET은 canonical sequence보다 set frontier를 autoregress한다는 점에서 일반 sequence AR과 다릅니다.

### [[concepts/generative-models/flow-matching|Flow Matching]]

Outer loop는 autoregressive지만 새 atom coordinate head는 Flow Matching입니다.

### [[concepts/architectures/set-transformer|Set Transformer]]

Permutation-invariant molecular context를 유지하는 backbone contract와 연결됩니다.

### [[concepts/generative-models/conditional-generation|Conditional generation]]

Cross-attention, adaptive normalization, CFG strength의 trade-off를 볼 수 있습니다.

### [[concepts/geometric-deep-learning/equivariance|Equivariance]]

Strict equivariant backbone 대신 rotation augmentation과 coordinate embedding으로 geometry를 다루는 다른 design point입니다.

### [[molecular-modeling/structure-based/index|Structure-Based Modeling]]

Pocket definition, static protein assumption, known-ligand pocket extraction boundary를 함께 봐야 합니다.

### [[concepts/sbdd/pose-quality|Pose quality]]

PB validity, clash, strain은 서로 다른 pose-quality diagnostics입니다.

### [[concepts/sbdd/binding-affinity|Binding affinity]]

Vina는 docking proxy이며 prospective affinity evidence가 아닙니다.

---

## 27. Final verdict

**Verdict: Must Read for fast pocket-conditioned generation and fragment completion.**

NEAT-POCKET의 가장 중요한 메시지는 “autoregressive model이 diffusion보다 좋다”가 아닙니다.

더 정확한 메시지는 다음입니다.

> **3D molecular generation의 global iterative trajectory를 molecule construction order와 local coordinate-generation subproblem으로 분해하면, quality–compute frontier를 크게 바꿀 수 있다.**

또 fragment completion에서는 architecture와 task interface가 잘 맞습니다.

- fixed fragment는 이미 존재하는 state
- 새 chemistry만 autoregress
- prefix를 다시 denoise할 필요 없음

반면 evidence boundary도 분명합니다.

- docking/pose benchmark이지 experimental hit discovery가 아님
- raw docking은 size confound가 큼
- static known pocket setting
- strong CFG에는 physical-validity trade-off
- large molecule에서 AR error accumulation은 아직 중요한 질문

따라서 후속 연구에서 가장 가치 있는 검증은 “leaderboard score 하나”가 아니라 **matched compute에서 global iterative generation과 set-autoregressive local flow를 비교하는 것**입니다.

## 6개월 뒤 기억해야 할 세 가지

1. **NEAT-POCKET은 pure autoregressive coordinate regressor가 아니다.** Molecule은 atom-by-atom으로 만들지만 각 새 atom의 3D 위치는 local Flow Matching head가 생성한다.
2. **Pocket condition은 fine→coarse→fine encoder + cross-attention + global adaptive normalization으로 들어가며, 새 conditioning path를 zero-init해 pretrained molecular prior를 보존한다.**
3. **가장 실용적인 architecture advantage는 fragment completion이다.** Prefix를 inpainting constraint가 아니라 initial atom set으로 두기 때문에 exact fragment preservation이 construction rule이 되고, 대신 prefix가 커질수록 diversity가 감소하는 trade-off를 명확히 측정해야 한다.

## Sources

- [Primary paper — arXiv:2609.05097](https://arxiv.org/abs/2609.05097)
- [Official NEAT-POCKET implementation](https://github.com/molinfo-vienna/NEAT-POCKET)
- [Reviewed official code snapshot — 323f262](https://github.com/molinfo-vienna/NEAT-POCKET/tree/323f262b17e1ee03f18e8dbaffe31088c6df2993)
- [Released model weights — Figshare](https://doi.org/10.6084/m9.figshare.33426877)
- [Official training configuration](https://github.com/molinfo-vienna/NEAT-POCKET/blob/323f262b17e1ee03f18e8dbaffe31088c6df2993/scripts/config_files/config_training.yaml)
- [Official evaluation script](https://github.com/molinfo-vienna/NEAT-POCKET/blob/323f262b17e1ee03f18e8dbaffe31088c6df2993/scripts/evaluation.py)
- [Official NEAT-POCKET model implementation](https://github.com/molinfo-vienna/NEAT-POCKET/blob/323f262b17e1ee03f18e8dbaffe31088c6df2993/src/neat/model/neat.py)
