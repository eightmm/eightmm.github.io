---
title: SurfSpec — Enhancing Off-Target-Agnostic Specificity by Bounding Pocket-Ligand Geometric Mismatch
aliases:
  - papers/surfspec
  - papers/sbdd/surfspec
tags:
  - papers
  - sbdd
  - structure-based-modeling
  - lead-optimization
  - molecular-generation
  - specificity
  - diffusion
status: full-note
source_type: ArXiv
source_url: https://arxiv.org/abs/2609.02963
---

# SurfSpec: Enhancing Off-Target-Agnostic Specificity by Bounding Pocket-Ligand Geometric Mismatch

> **한 줄 요약:** SurfSpec은 알려진 off-target 구조를 optimization input으로 쓰지 않고, **target pocket–ligand surface mismatch를 metric으로 만든 뒤 triangle inequality와 empirical geometry–affinity calibration을 연결해 geometrically separated off-target class에 대한 conservative specificity lower bound를 유도**하고, 그 분석을 surface-directed ligand growth + diffusion refinement로 구현합니다.

## 왜 이 논문을 저장하는가

Structure-based lead optimization에서는 흔히 “target에 더 잘 붙는 molecule”을 만들면 충분하다고 생각하기 쉽습니다. 하지만 target affinity와 specificity는 같은 objective가 아닙니다.

어떤 ligand $L$이 target $P_{\mathrm{tgt}}$에 강하게 결합하더라도, 여러 off-target $O$에도 같이 강하게 결합하면 실제 selectivity는 낮을 수 있습니다. 논문이 사용하는 추상화는 다음과 같습니다.

$$
\operatorname{Spec}(L;P_{\mathrm{tgt}},\mathcal O)
=
A_{P_{\mathrm{tgt}}}(L)
-
\max_{O\in\mathcal O} A_O(L).
$$

즉 specificity는 **target affinity 자체**가 아니라 target과 strongest off-target 사이의 margin입니다.

기존 specificity-aware 방법의 자연스러운 접근은 off-target structure나 activity label을 직접 주는 것입니다. 문제는 실제 lead optimization 초기에 가능한 off-target set이 완전하지 않다는 점입니다. SurfSpec은 이 조건을 더 어렵게 잡습니다.

> **Optimization 중에는 target pocket만 본다. 그럼에도 어떤 종류의 off-target에 대해서는 specificity-like guarantee를 만들 수 있는가?**

이 질문 때문에 이 논문은 generator architecture 자체보다 **evaluation contract와 geometric certificate**가 더 오래 남을 가치가 있습니다.

---

## Metadata

| Field | Value |
| --- | --- |
| Paper | SurfSpec: Enhancing Off-Target-Agnostic Specificity by Bounding Pocket-Ligand Geometric Mismatch |
| Authors | Minyeong Hwang, Yoorim Gang, Ziseok Lee, Wooyeol Lee, Young Bin Park, Jae-Mun Choi, Kyungsu Kim, Eunho Yang |
| Version | arXiv v1 |
| Submitted | 2026-09-02 |
| arXiv | [2609.02963](https://arxiv.org/abs/2609.02963) |
| Main benchmark | CrossDocked2020, 100 test complexes |
| Growth model | DiffLinker |
| Pocket-conditioned prior | DiffSBDD |
| Main docking proxy | AutoDock Vina |
| Official code/checkpoint | **to verify** — no official release was verified as of 2026-09-05 |

---

## 1. Problem: affinity maximization은 specificity optimization이 아니다

Lead optimization의 최소한 세 축을 분리해야 합니다.

$$
\underbrace{\text{target binding}}_{\text{affinity}}
\qquad
\underbrace{\text{avoid other pockets}}_{\text{specificity}}
\qquad
\underbrace{\text{chemical / structural validity}}_{\text{feasibility}}.
$$

이 세 가지는 서로 상관될 수 있지만 동일하지 않습니다.

예를 들어 ligand size를 크게 늘리면 target pocket contact가 많아져 docking score가 좋아질 수 있습니다. 동시에 다른 pockets에도 promiscuously 맞거나 steric clash가 커질 수 있습니다. 따라서 다음 implication은 성립하지 않습니다.

$$
\Delta A_{\mathrm{target}}>0
\;\not\Rightarrow\;
\Delta \operatorname{Spec}>0.
$$

SurfSpec의 핵심 문제 설정은 **off-target-agnostic lead optimization**입니다. Optimization algorithm은 $P_{\mathrm{tgt}}$만 접근하고 evaluation에서만 held-out off-target pockets를 사용합니다.

이 setting은 현실적인 장점이 있지만 동시에 claim boundary를 명확히 해야 합니다. “off-target을 보지 않는다”는 것이 “모든 unknown off-target에 안전하다”는 뜻은 아닙니다. 이 논문의 guarantee는 **geometrically separated off-target class**에 한정됩니다.

---

## 2. 핵심 아이디어: surface mismatch를 metric으로 만든다

SurfSpec의 이론은 pocket과 ligand를 단순한 point cloud similarity score로 비교하는 것이 아니라, surface에서 유도된 probability measure 사이의 distance로 바꿉니다.

Ligand의 heavy-atom van der Waals surface를 $S_L$이라 하고, ligand 기준으로 정렬된 pocket surface 중 ligand heavy atom에서 일정 거리 안에 있는 부분을 $S_P^L$라고 둡니다. 논문의 구현에서는 pocket surface를 ligand heavy atom으로부터 **8 Å 이내**로 제한합니다.

두 surface는 ligand-local domain 위의 signed-distance field를 사용해 Boltzmann-like probability density로 변환됩니다.

$$
S_L \rightarrow \mu_L,
\qquad
S_P^L \rightarrow \nu_P^L.
$$

그 다음 geometric mismatch를 square-root Jensen–Shannon divergence로 정의합니다.

$$
d_{\mathrm{gm}}(L,P)
=
d_{\mathrm{JS}}(\mu_L,\nu_P^L).
$$

여기서 중요한 선택은 단순 JS divergence가 아니라 **Jensen–Shannon distance**, 즉 square-root 형태를 사용한다는 점입니다. 이 값은 metric이므로 triangle inequality를 만족합니다.

$$
d(a,c) \le d(a,b)+d(b,c).
$$

이 한 가지 성질이 specificity certificate의 핵심입니다.

### 왜 surface probability measure인가

단순 atom-to-atom distance나 volume overlap만으로 pocket fit을 정의하면 서로 다른 atom count, local surface density, ligand size 변화에 민감할 수 있습니다. Surface-induced distribution은 “현재 ligand가 target pocket surface를 얼마나 채우고 있는가”를 비교 가능한 형태로 만듭니다.

다만 이것 역시 physical binding energy가 아닙니다.

$$
d_{\mathrm{gm}}
\neq
\Delta G
\neq
K_d
\neq
\text{experimental selectivity}.
$$

SurfSpec은 geometry를 specificity로 바로 동일시하지 않고 다음 단계에서 empirical calibration을 붙입니다.

---

## 3. Geometry–affinity calibration은 theorem이 아니라 empirical assumption이다

논문은 CrossDocked2020의 reference ligand–pocket pairs에서 geometric mismatch와 docking-based affinity 사이의 관계를 추정합니다.

먼저 training split의 $n=531$ examples에서 linear trend를 fit합니다.

$$
\hat a(d)=\beta_0+\beta_1 d.
$$

그 다음 residual의 empirical quantile을 사용해 mismatch $d$에 대한 affinity envelope를 만듭니다.

$$
f_-^\eta(d)
\le
A_P(L)
\le
f_+^\eta(d).
$$

여기서 $\eta$는 tail probability이고, paper는 90%, 95%, 99% 수준의 empirical interval을 확인합니다.

중요한 점은 이 calibration이 물리 법칙이 아니라는 것입니다. Paper에서도 이를 empirical assumption으로 취급합니다. 보고된 held-out coverage는 대략 91%, 92%, 97% 수준으로 calibration이 어느 정도 맞지만, 이것은 CrossDocked distribution과 Vina-based affinity proxy에 대한 결과입니다.

따라서 가장 안전한 해석은:

> **geometric mismatch가 affinity와 충분히 monotonic한 benchmark regime에서는 mismatch bound를 affinity margin의 conservative proxy로 옮길 수 있다.**

이지,

> **surface mismatch만 알면 실제 binding affinity를 안다**

가 아닙니다.

---

## 4. Triangle inequality가 off-target lower bound를 만드는 방법

Target pocket과 off-target pocket이 ligand-oriented geometry에서 충분히 떨어져 있다고 합시다.

$$
m_L(O)
=
d_{\mathrm{JS}}
\left(
\nu_{P_{\mathrm{tgt}}}^L,
\nu_O^L
\right).
$$

Off-target class $\mathcal O_\delta$가 $\delta$-separated라는 것은 모든 relevant off-target에 대해 대략

$$
m_L(O)\ge \delta
$$

가 유지된다는 뜻입니다.

현재 ligand의 target mismatch를

$$
\epsilon
=
d_{\mathrm{gm}}(L,P_{\mathrm{tgt}})
$$

라고 하면 triangle inequality로

$$
d_{\mathrm{gm}}(L,O)
\ge
[\delta-\epsilon]_+
$$

를 얻습니다.

즉 target mismatch $\epsilon$을 줄이면 geometrically separated off-target까지 동시에 잘 맞기 어려워지는 방향의 lower bound가 생깁니다.

이를 empirical affinity envelope와 연결하면 논문의 specificity lower bound는 개념적으로 다음 형태가 됩니다.

$$
\operatorname{Spec}(L;P_{\mathrm{tgt}},\mathcal O_\delta)
\gtrsim
f_-^\eta(\epsilon)
-
f_+^\eta([\delta-\epsilon]_+).
$$

Paper의 theorem은 finite off-target set에 대한 probability term까지 포함하며, confidence는 $|\mathcal O|$와 $\eta$에 의존합니다.

### 이 theorem의 진짜 의미

이론의 가치는 “geometry로 selectivity를 완전히 해결했다”가 아닙니다.

더 정확하게는:

1. mismatch를 proper metric으로 설계한다.
2. target과 off-target가 그 metric에서 충분히 떨어져 있다는 조건을 둔다.
3. target mismatch를 줄이면 off-target mismatch의 lower bound가 커지는 것을 triangle inequality로 보인다.
4. empirical geometry–affinity calibration이 성립하는 범위에서 이를 affinity margin으로 번역한다.

즉 **assumption이 어디에 들어가는지 명확한 certificate**입니다.

이 구조는 다른 SBDD objective를 설계할 때도 재사용할 수 있습니다. “좋아 보이는 proxy”를 그냥 optimize하는 대신, proxy가 downstream utility와 연결되는 조건을 명시할 수 있기 때문입니다.

---

## 5. Certificate는 언제 vacuous해지는가

가장 중요한 caveat입니다.

Target과 off-target pocket이 geometric metric에서 충분히 멀지 않으면

$$
\delta-\epsilon \le 0
$$

가 되어 lower bound는 거의 정보를 주지 못합니다.

즉 structurally similar binding sites, homologous proteins, conserved ATP pockets처럼 **실제로 selectivity가 가장 어려운 경우**에 certificate가 약해질 수 있습니다.

Paper appendix에서는 random target–off-target pairs에서

$$
M_L(O)
=
m_L(O)-d_{\mathrm{gm}}(L,P_{\mathrm{tgt}})
$$

를 계산하고, filtered usable pairs 중 **65.96%**가 positive margin을 가진다고 보고합니다. 이것은 certificate가 항상 vacuous하지는 않다는 evidence입니다.

하지만 random CrossDocked pocket pairs는 실제 medicinal chemistry의 hardest off-target panel과 다릅니다. 그래서 이 숫자는 “65.96%의 실제 off-target에 안전하다”로 읽으면 안 됩니다.

더 좋은 prospective test는 target의 close paralogs, structurally similar pockets, known pharmacological anti-targets를 일부러 선택해 margin distribution이 어떻게 무너지는지 보는 것입니다.

---

## 6. SurfSpec pipeline: theory를 실제 optimization으로 바꾸기

SurfSpec의 generation loop는 다음처럼 읽는 것이 가장 명확합니다.

```text
initial lead + target pocket
    ↓
target-pocket surface occupancy 분석
    ↓
가장 가까운 feasible under-occupied surface patch 선택
    ↓
DiffLinker로 해당 patch 방향 ligand growth
    ↓
pocket clash가 큰 atom 제거 → geometric pseudo-label
    ↓
DiffSBDD pocket-conditioned prior 아래 low-noise recovery
    ↓
valid refined ligand
    ↓
repeat, 최대 K = 3
```

이 구조에서 theory와 generator는 역할이 분리되어 있습니다.

- **Theory / geometry:** 어디를 채워야 target mismatch가 줄어드는지 제안합니다.
- **Linker generator:** 그 방향으로 atom-level growth proposal을 만듭니다.
- **Pocket-conditioned diffusion prior:** geometric pseudo-label을 chemically/structurally plausible ligand distribution으로 되돌립니다.

즉 SurfSpec은 end-to-end 새 foundation model보다 **existing generative components를 geometry-driven optimization loop로 묶은 method**에 가깝습니다.

---

## 7. Surface patch selection: 무작정 ligand를 키우는 것이 아니다

각 iteration에서 current ligand가 충분히 점유하지 않은 target surface patch를 찾습니다. Shared benchmark setting에서는 ligand와 patch 사이의 feasible distance를 대략 **4–10 Å** 범위로 제한합니다.

그 다음 DiffLinker가 current ligand와 selected surface region을 향하는 linker/growth proposal을 만듭니다.

중요한 것은 size extension baseline과의 차이입니다.

단순히 atom을 많이 붙이면 target pocket surface와 overlap이 늘어 geometric mismatch가 작아질 수 있습니다. 하지만 그 molecule은 steric clash가 심하거나 다른 pockets에도 잘 맞을 수 있습니다.

SurfSpec은:

$$
\text{more atoms}
$$

가 아니라

$$
\text{under-occupied target surface를 향한 controlled growth}
$$

를 objective로 둡니다.

Pseudo-label을 만들 때 target pocket atom과 **2 Å 미만**으로 clash하는 growth atom을 잘라냅니다. 이 단계는 chemical validity를 완전히 해결하지 않으므로 다음 refinement가 필요합니다.

---

## 8. Pseudo-label refinement: geometry와 molecular prior 사이의 reconciliation

Surface-directed growth로 얻은 pseudo-label $x_{\mathrm{label}}$은 원하는 geometry를 담지만 pretrained molecular distribution 밖에 있을 수 있습니다.

SurfSpec은 pocket-conditioned clean ligand prior $p_0(x\mid P_{\mathrm{tgt}})$와 anchor penalty를 결합한 localized distribution을 생각합니다.

$$
p_0^{\mathrm{anc}}
(x\mid P_{\mathrm{tgt}},x_{\mathrm{label}})
\propto
p_0(x\mid P_{\mathrm{tgt}})
\exp
\left(
-\frac{\lambda}{2}
\|x-x_{\mathrm{label}}\|^2
\right).
$$

여기서 두 힘이 경쟁합니다.

- prior term: valid pocket-conditioned ligand처럼 보이게 한다.
- anchor term: geometric pseudo-label에서 너무 멀어지지 않게 한다.

이것은 generic posterior sampling 문제로 풀 수도 있지만 SurfSpec은 더 단순한 task-specific route를 선택합니다.

---

## 9. 왜 low-noise recovery인가

Authors는 이 pseudo-label이 완전히 arbitrary observation이 아니라 이미 원하는 ligand 근처에 있는 **localized geometric proposal**이라고 봅니다.

그래서 high-noise regime에서 posterior를 계속 estimate하기보다, 한 intermediate low-noise level에서 anchored mode를 찾고 그 주변에서 sampling을 마칩니다.

공개 paper configuration에서 recovery noise time은

$$
\tau=0.15
$$

입니다.

개념적으로:

1. low-noise anchored score flow로 mode neighborhood를 찾습니다.
2. recovered point 주변에 다시 작은 noise를 넣습니다.
3. guided reverse SDE로 clean sample까지 갑니다.

Authors 스스로 이 component를 **new general-purpose inverse solver**라기보다 surface-directed ligand growth에 맞춘 practical recovery method로 제한합니다. 이 claim restraint는 중요합니다.

---

## 10. Evaluation contract: 무엇을 실제로 측정했는가

Main benchmark는 **CrossDocked2020 test complexes 100개**입니다.

각 target에서 reference ligand를 initial lead로 사용하고 각 method가 optimized molecule 하나를 만듭니다.

Off-target evaluation은 각 target마다 나머지 99 test pockets 중 **10개를 random sampling**해 $\mathcal O_{\mathrm{eval}}$을 구성합니다.

여기서 실험 설계를 구분해야 합니다.

- Off-target-agnostic methods는 optimization 중 $\mathcal O_{\mathrm{eval}}$을 보지 않습니다.
- ActivityDiff는 true evaluation off-target가 아니라 별도 random surrogate negatives를 사용합니다.
- Final empirical specificity는 target/off-target AutoDock Vina score에서 계산됩니다.

따라서 example unit은 “target complex + optimized ligand”이지만 specificity evaluation은 **randomly sampled CrossDocked pockets**에 의존합니다.

이것은 실제 known off-target assay panel과 동일한 generalization claim이 아닙니다.

---

## 11. Metric을 하나로 합치면 안 된다

SurfSpec paper를 읽을 때 최소 다음 metric family를 분리해야 합니다.

| Axis | Metric / proxy | 실제 질문 |
| --- | --- | --- |
| Target binding | AutoDock Vina | target docking proxy가 좋아지는가? |
| Empirical specificity | target-vs-random-off-target Vina margin | held-out random pockets보다 target을 선호하는가? |
| Geometry | $d_{\mathrm{gm}}$ | target surface mismatch가 줄었는가? |
| Pocket coverage | occupancy | target pocket surface를 더 채우는가? |
| Validity | pocket clash rate | obvious steric invalidity가 생기는가? |
| Size | heavy atom count | gain이 단순 size 증가에서 오는가? |
| Refinement | valence / bond MMD / repulsion | prior consistency가 유지되는가? |
| Anchor faithfulness | RMSD / topology similarity | pseudo-label intent를 얼마나 보존하는가? |

특히 다음 세 식을 분리해서 기억하는 것이 좋습니다.

$$
\text{lower mismatch}
\not\Rightarrow
\text{higher specificity},
$$

$$
\text{better Vina}
\not\Rightarrow
\text{experimental affinity},
$$

$$
\text{random-pocket specificity}
\not\Rightarrow
\text{clinical off-target safety}.
$$

---

## 12. Main result: SurfSpec이 실제로 잘한 것은 무엇인가

Table 1의 핵심 결과를 decision-useful하게만 요약하면 다음과 같습니다.

| Method | Empirical specificity avg ↑ | Mismatch avg ↓ | Occupancy avg ↑ | Target Vina avg ↓ | Clash rate ↓ | # atoms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial lead | -0.83 | 0.540 | 0.26 | -7.17 | 0.00 | 22.75 |
| PMDM | -1.16 | 0.501 | 0.33 | -6.47 | 0.16 | 31.00 |
| ActivityDiff | -0.89 | 0.538 | 0.27 | -7.07 | 0.11 | 22.89 |
| DiffSBDD SizeExt +20 | -1.71 | 0.480 | 0.28 | -7.57 | 0.45 | 35.88 |
| DiffSBDD SizeExt +30 | -1.81 | **0.468** | 0.31 | -7.14 | 0.62 | 41.58 |
| **SurfSpec** | **-0.71** | 0.480 | **0.36** | -8.63 | **0.00** | 34.99 |

여기서 가장 좋은 해석은:

1. SurfSpec은 tested methods 중 **best average empirical specificity**를 보고합니다.
2. pocket occupancy도 가장 높고 clash는 0입니다.
3. target Vina는 강하지만 모든 baseline 중 unique best는 아닙니다.
4. SurfSpec의 mismatch가 가장 낮은 것도 아닙니다.
5. 오히려 naive size extension이 mismatch를 더 낮추지만 clash와 specificity가 크게 악화됩니다.

이 마지막 point가 중요합니다.

> **mismatch는 objective direction을 제공하지만, mismatch를 무조건 최소화한다고 specificity가 자동으로 좋아지는 것은 아니다.**

실제 usable optimization에는 controlled growth와 learned ligand prior가 같이 필요합니다.

---

## 13. Thresholded specificity도 같은 방향인가

Paper는 average margin뿐 아니라 empirical specificity가 일정 threshold를 넘는 비율을 보고합니다.

SurfSpec의 success rate는 대략:

- $>0.2$: **0.18**
- $>0.4$: **0.16**
- $>0.6$: **0.15**

로 tested methods 중 가장 높습니다.

절대값만 보면 여전히 낮습니다. 즉 random off-target evaluation에서도 대부분 sample이 large positive specificity margin을 가지는 것은 아닙니다.

따라서 paper의 claim은 “selective molecule generation을 해결했다”가 아니라:

> **동일 benchmark setting에서 off-target을 optimization 중 보지 않으면서 empirical target-over-off-target preference를 개선했다.**

정도가 적절합니다.

---

## 14. Size-extension baseline이 매우 중요한 negative control이다

SurfSpec의 가장 유익한 control 중 하나는 DiffSBDD-based size extension입니다.

SizeExt +30은 mismatch를 0.468까지 낮춰 SurfSpec의 0.480보다 낮습니다. 그런데 clash rate는 0.62이고 empirical specificity는 -1.81로 오히려 크게 나빠집니다.

즉 다음 단순 가설은 falsified됩니다.

$$
\text{ligand를 크게 만들어 pocket을 많이 채우면 specificity도 좋아진다}.
$$

이 결과는 geometric mismatch metric 자체를 비판적으로 읽게 합니다.

- mismatch는 surface occupancy와 강하게 연결될 수 있습니다.
- 하지만 molecule validity와 steric feasibility를 무시한 mismatch optimization은 잘못된 direction으로 갈 수 있습니다.
- specificity certificate의 practical usefulness는 **reachable valid ligand set** 안에서 mismatch를 줄이는 optimization에 달려 있습니다.

따라서 SurfSpec의 핵심은 metric 하나보다:

$$
\boxed{
\text{geometric target}
+
\text{controlled growth}
+
\text{molecular prior recovery}
}
$$

의 결합입니다.

---

## 15. Refinement ablation: faithfulness만 최대화하는 것도 답이 아니다

Table 2는 pseudo-label recovery를 여러 inverse/refinement method와 비교합니다.

SurfSpec recovery는 다음 trade-off를 보입니다.

- pocket clash $<2$ Å: **0.01**
- valence validity: **1.00**
- target-affinity diagnostic global: **8.00**
- pseudo-label RMSD avg: **0.78**
- topological similarity avg: **0.48**

반면 SDEdit은 pseudo-label RMSD가 약 0.31로 훨씬 낮고, Red-Diff는 topology similarity가 매우 높습니다.

그렇지만 SurfSpec은 prior consistency와 geometric intent 사이에서 더 균형 잡힌 결과를 보입니다.

이것은 generative refinement의 중요한 일반 원칙을 보여줍니다.

> **Anchor를 정확히 복원하는 것과 usable molecule을 만드는 것은 같은 objective가 아니다.**

Pseudo-label이 이미 chemically invalid할 수 있기 때문에 fidelity-only solver는 오히려 잘못된 constraint를 충실히 보존할 수 있습니다.

---

## 16. What is actually new?

SurfSpec을 구성 요소별로 분해하면 novelty가 더 명확해집니다.

### Diffusion prior 자체

DiffSBDD를 pretrained pocket-conditioned prior로 사용하므로 diffusion model 자체가 novelty는 아닙니다.

### Linker generation 자체

DiffLinker 기반 growth도 독립적인 novelty는 아닙니다.

### Surface complementarity 자체

Pocket–ligand shape complementarity는 classical docking부터 오래된 아이디어입니다.

### 진짜 핵심 1: metricized mismatch → certificate

Surface complementarity를 proper metric으로 정의하고 triangle inequality를 통해 **separated off-target class에 대한 target-only bound**로 연결한 것이 가장 강한 conceptual contribution입니다.

### 진짜 핵심 2: certificate의 practical operationalization

Theorem을 그대로 optimizer로 풀기보다 under-occupied surface patch growth라는 simple design principle로 바꾸고, pretrained prior recovery로 feasibility를 유지합니다.

따라서 이 논문을 기억할 때는:

> “Diffusion lead optimization paper”

보다

> **“target-only geometry를 specificity surrogate로 만들 때 assumption과 certificate를 명시한 paper”**

로 기억하는 편이 더 오래 갑니다.

---

## 17. Evidence가 실제로 지지하는 claim

현재 evidence로 비교적 강하게 말할 수 있는 것은 다음입니다.

### 17.1 Random held-out CrossDock pockets에서 empirical specificity가 개선된다

Optimization 중 사용하지 않은 random off-target pockets를 evaluation에만 사용했을 때 SurfSpec이 tested off-target-agnostic baselines보다 target preference를 높입니다.

### 17.2 Controlled surface growth는 naive size growth보다 낫다

Size extension은 mismatch를 낮춰도 clash와 specificity가 악화될 수 있고, SurfSpec은 높은 occupancy와 낮은 clash를 함께 달성합니다.

### 17.3 Low-noise recovery는 pseudo-label fidelity와 prior consistency의 practical compromise를 만든다

Refinement table에서 exact anchor preservation보다 validity/repulsion/target-affinity diagnostics의 균형이 더 좋습니다.

### 17.4 Certificate가 완전히 vacuous하지만은 않다

Random CrossDock target/off-target pairs의 substantial fraction에서 positive geometric margin을 관찰합니다.

---

## 18. 이 논문이 아직 증명하지 못한 것

### 18.1 Experimental selectivity

AutoDock Vina margin은 experimental $K_d$, $K_i$, $IC_{50}$, functional selectivity가 아닙니다.

$$
\operatorname{Spec}_{\mathrm{Vina}}
\neq
\operatorname{Spec}_{\mathrm{assay}}.
$$

### 18.2 Unknown off-target safety

Off-target을 optimization에서 보지 않았다는 사실은 open-world toxicology coverage를 의미하지 않습니다.

### 18.3 Closely related pockets

Theorem 자체가 geometric separation을 요구하므로 hard negative pockets에서 가장 먼저 약해질 수 있습니다.

### 18.4 Strong OOD generalization

100 CrossDocked test complexes와 같은 test pool에서 random off-target을 뽑은 평가는 protein-family OOD, scaffold OOD, temporal OOD와 다릅니다.

### 18.5 Mismatch가 causal mechanism이라는 것

SurfSpec pipeline 전체가 좋아진 결과만으로 geometric mismatch 자체가 gain의 유일한 원인이라고 말할 수 없습니다. Patch selection, linker prior, stopping rule, DiffSBDD recovery가 모두 기여할 수 있습니다.

---

## 19. Evaluation에서 가장 신경 써야 할 selection artifact

Main Vina evaluation에는 docking timeout과 failure handling이 포함됩니다.

Paper의 main analysis는 AutoDock Vina run에 timeout을 두고, docking failure는 해당 aggregation에서 제외하는 protocol을 사용합니다. Target docking failure는 specificity aggregation에서도 빠질 수 있습니다.

이것은 중요한 evaluation risk입니다.

어떤 method가 harder-to-dock molecule을 더 많이 만들면 단순히 successful subset만 비교해서 score가 좋아 보일 수 있습니다.

따라서 더 강한 report는 최소 다음을 같이 내야 합니다.

$$
\text{metric among successes}
+
\text{docking success rate}.
$$

Appendix의 relaxed-timeout/resampled evaluation은 이 문제를 일부 확인하지만, future benchmark에서는 failure를 missing data가 아니라 method output의 일부로 취급하는 편이 더 안전합니다.

---

## 20. 내가 재현한다면 가장 먼저 할 5-arm ablation

Architecture보다 objective attribution을 먼저 분리하겠습니다.

| Arm | Growth signal | Prior recovery | 목적 |
| --- | --- | --- | --- |
| A | none | none | initial lead |
| B | size-only | DiffSBDD | 단순 ligand enlargement |
| C | occupancy-only | DiffSBDD | coverage 효과 |
| D | mismatch-directed | generic low-noise edit | metric 효과 |
| E | full SurfSpec | SurfSpec recovery | full system |

모든 arm은 다음을 동일하게 맞춰야 합니다.

- initial ligand
- maximum heavy-atom budget
- number of optimization iterations
- diffusion/sample budget
- Vina evaluation budget
- random seed count
- off-target panel
- docking failure policy

그렇지 않으면 “specificity gain”이 사실상 더 많은 atoms, 더 많은 compute, 더 쉬운 docking subset에서 온 것인지 분리하기 어렵습니다.

---

## 21. 더 강한 off-target benchmark는 어떻게 만들 것인가

Random CrossDock pockets는 first test로는 괜찮지만 specificity 연구의 최종 benchmark로는 약합니다.

나는 off-target panel을 최소 세 단계로 나누는 것이 좋다고 봅니다.

### Tier 1 — Random structural negatives

Paper와 비슷한 random pockets. 시스템의 coarse preference를 확인합니다.

### Tier 2 — Geometrically close hard negatives

Target pocket과 shape/embedding/structure similarity가 높은 pockets만 선택합니다.

이 setting이 theorem의 $\delta$-separation assumption을 직접 stress-test합니다.

### Tier 3 — Known pharmacological off-targets

가능하면 실제 assay/selectivity information이 있는 target family를 사용합니다.

여기서는 docking margin보다 experimental or high-quality activity label을 primary endpoint로 둡니다.

이 세 tier를 함께 보면:

$$
\text{certificate가 언제 유효하고 언제 무너지는가}
$$

를 훨씬 명확하게 볼 수 있습니다.

---

## 22. OOD split contract

Strong generalization claim을 하려면 random complex split을 넘어서야 합니다.

### Ligand OOD

Bemis–Murcko scaffold가 train/test를 넘지 않게 합니다.

### Protein OOD

Protein sequence 또는 structural family cluster를 split unit으로 사용합니다.

### Pocket OOD

Pocket similarity metric으로 cluster를 만들고 near-identical pocket leakage를 막습니다.

### Joint OOD

가장 강한 setting은:

$$
\text{new protein/pocket family}
+
\text{new ligand scaffold}.
$$

SurfSpec의 target-only geometric principle이 benchmark chemistry prior보다 genuinely reusable하다면 이 setting에서도 상대적 이득이 남아야 합니다.

---

## 23. Specificity certificate를 falsify하는 실험

이론적 contribution을 존중하려면 “잘 되는 예”보다 **깨지는 조건**을 적극적으로 찾는 것이 좋습니다.

가장 직접적인 test는 target과 off-target pocket similarity를 연속적으로 올리는 것입니다.

$$
\delta_1 > \delta_2 > \cdots > \delta_k.
$$

각 bin에서:

- positive certificate margin fraction
- target mismatch
- empirical Vina specificity
- experimental selectivity가 있다면 assay gap
- SurfSpec gain over affinity-only baseline

을 측정합니다.

좋은 이론이라면 “$\delta$가 작아질수록 certificate와 empirical gain이 약해진다”는 predictable failure curve를 보여야 합니다.

오히려 이런 failure map이 있으면 deployment boundary가 더 명확해집니다.

---

## 24. Mismatch metric 자체의 필요한 ablation

Jensen–Shannon distance를 선택한 이유는 triangle inequality를 쓸 수 있다는 점이 강합니다. 하지만 predictive utility가 JS 자체에서 오는지 확인해야 합니다.

비교할 수 있는 controls:

- surface Chamfer distance
- signed-distance-field $L_2$
- Wasserstein distance
- volume overlap / IoU-like score
- learned pocket–ligand geometric embedding distance
- JS divergence without square-root
- Jensen–Shannon distance

두 축으로 평가해야 합니다.

1. **certificate mathematics:** metric / triangle inequality를 쓸 수 있는가?
2. **empirical utility:** affinity/selectivity proxy와 얼마나 잘 correlate하는가?

수학적으로 예쁜 metric이 biological utility에 가장 좋은 metric이라고 가정해서는 안 됩니다.

---

## 25. Representation / coordinate contract

SurfSpec은 coordinate-heavy pipeline이므로 frame과 preprocessing을 분명히 해야 합니다.

### Input object

- ligand heavy atoms + coordinates
- target pocket atoms + coordinates
- target pocket surface
- reference/current lead

### Transformations

Rigid-body alignment이 theory에 명시적으로 들어갑니다. 따라서 global translation/rotation 자체가 mismatch를 바꾸면 안 됩니다.

### Derived representation

- vdW surface
- ligand-oriented pocket surface
- signed-distance field
- normalized surface probability measure

### Leakage risk

Evaluation이나 preprocessing에서 reference ligand pose가 target pocket definition이나 alignment에 deployment-unavailable한 방식으로 사용된다면 leakage가 될 수 있습니다.

Paper setting은 reference lead가 주어지는 **lead optimization**이므로 initial ligand pose availability 자체는 task definition의 일부입니다. 그러나 de novo generation setting으로 옮길 때는 이 assumption을 그대로 가져가면 안 됩니다.

---

## 26. Reproducibility status

현재 paper는 method/configuration을 상당히 자세히 설명하지만, 2026-09-05 기준 이 review에서 **SurfSpec 공식 code/checkpoint release를 검증하지 못했습니다.**

따라서 Reproducibility를 높게 평가하기는 어렵습니다.

재현에 필요한 최소 요소는:

- CrossDocked2020 exact 100-complex subset
- target/off-target sampling seed
- pocket surface construction
- JS mismatch implementation
- geometry–affinity calibration split
- DiffLinker checkpoint/config
- DiffSBDD checkpoint/config
- patch feasibility rule
- pseudo-label truncation rule
- low-noise recovery hyperparameters
- Vina executable/config/timeout
- docking failure policy

입니다.

Paper appendix가 여러 값을 제공하더라도, exact pipeline script와 processed split이 공개되지 않으면 hidden degrees of freedom이 남습니다.

---

## 27. Practical reproduction path

공식 implementation이 공개되기 전이라면 full reproduction보다 다음 순서가 안전합니다.

1. **Metric-only reproduction**
   CrossDock reference pairs에서 surface measure와 $d_{\mathrm{gm}}$을 구현하고 geometry–Vina calibration curve를 재현합니다.

2. **Certificate analysis**
   Random off-target pairs와 similarity-stratified pairs에서 positive margin fraction을 확인합니다.

3. **Naive growth controls**
   ligand size extension이 mismatch를 낮추지만 clash/specificity를 악화시키는지 확인합니다.

4. **Surface-directed growth without custom recovery**
   DiffLinker + standard DiffSBDD edit로 geometry signal 자체의 value를 분리합니다.

5. **Full low-noise recovery**
   마지막에 SurfSpec-specific recovery를 추가해 attribution을 봅니다.

이 순서면 full system이 안 맞을 때 어느 component가 문제인지 알 수 있습니다.

---

## 28. Failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| Random off-target에서 좋아진 것을 open-world safety로 해석 | evaluation population이 완전히 다름 |
| Vina gap을 experimental selectivity로 해석 | docking proxy bias |
| target affinity improvement를 specificity로 해석 | strongest off-target margin이 빠짐 |
| mismatch 감소만 보고 성공 판단 | size extension control이 반례 |
| hard negative pocket을 빼고 certificate를 일반화 | $\delta$-separation이 핵심 assumption |
| docking failure sample을 제외 | selection bias 가능성 |
| atom count budget을 맞추지 않음 | larger ligand advantage confound |
| reference lead availability를 de novo setting에 그대로 적용 | task-definition leakage |
| calibration fit/test boundary를 섞음 | certificate confidence 과대평가 |
| one seed off-target sampling | random negative set variance |

---

## 29. LiFT와 비교하면 무엇이 다른가

[[papers/generative-models/lift|LiFT]]와 SurfSpec은 둘 다 SBDD molecular generation을 control하지만 control source가 다릅니다.

LiFT는 language/chemical semantic prior를 geometric flow에 condition으로 넣어 **어떤 chemical trend를 원하는가**를 control합니다.

SurfSpec은 target surface geometry에서 under-occupied region과 mismatch를 계산해 **어디를 더 채워야 하는가**를 control합니다.

따라서 두 방법은 경쟁 관계보다 orthogonal axis로 볼 수 있습니다.

$$
\text{semantic preference}
+
\text{target-only geometric specificity signal}
+
\text{3D validity prior}.
$$

장기적으로는 이 세 pathway를 분리한 controllable generator가 더 해석 가능할 수 있습니다.

---

## 30. 이 논문에서 가장 재사용 가능한 연구 원칙

### 원칙 1 — utility proxy를 metric으로 만들 수 있으면 triangle inequality를 활용할 수 있다

단순 learned score보다 metric structure가 있으면 unseen object class에 대한 bound를 만들 여지가 생깁니다.

### 원칙 2 — theorem과 optimizer를 분리해라

Certificate가 존재한다고 exact bound optimization을 할 필요는 없습니다. SurfSpec은 theory에서 얻은 direction을 simple surface growth heuristic으로 operationalize합니다.

### 원칙 3 — proxy optimization에는 adversarially simple baseline이 필요하다

Size extension처럼 “metric만 좋아지는” baseline을 반드시 넣어야 metric gaming을 발견할 수 있습니다.

### 원칙 4 — specificity는 target score가 아니라 contrastive evaluation이다

항상 target와 off-target population 정의를 같이 써야 합니다.

---

## 31. 가장 중요한 next experiment

이 논문을 실제 연구에 가져온다면 나는 새 generator를 만들기보다 **benchmark layer부터 도입**하겠습니다.

같은 optimized ligands에 대해:

1. target Vina / Gnina
2. pocket occupancy
3. SurfSpec geometric mismatch
4. PoseBusters / clash
5. random off-target gap
6. structurally similar hard-negative gap
7. scaffold novelty / diversity

를 동시에 계산합니다.

그 다음 affinity-only optimization이 실제로 어떤 축을 희생하는지 봅니다.

이 experiment는 SurfSpec architecture를 재현하지 않아도 바로 정보량이 높습니다. 특히 기존 generation/refinement model의 “better docking score”가 target specificity로 이어지는지 분리할 수 있습니다.

---

## 32. Claim–evidence boundary

| Claim | 현재 evidence | 판단 |
| --- | --- | --- |
| Target-only geometry로 specificity-like signal을 만들 수 있다 | CrossDock random held-out pockets에서 Vina margin 개선 | **Supported in benchmark scope** |
| Mismatch 감소가 항상 specificity를 높인다 | Size extension이 반례 | **Not supported** |
| Geometrically separated off-target에 lower bound가 있다 | Metric + triangle inequality + empirical calibration | **Supported under assumptions** |
| Similar pockets에도 guarantee가 강하다 | separation assumption이 깨짐 | **Not supported** |
| Experimental selectivity가 개선된다 | wet-lab evidence 없음 | **Not established** |
| Open-world off-target safety가 개선된다 | random CrossDock pockets only | **Not established** |
| Full pipeline reproducible | official code/checkpoint 미검증 | **Not yet** |

이 표가 SurfSpec을 읽을 때 가장 중요한 boundary입니다.

---

## 33. Final verdict

**Verdict: Must Read for SBDD evaluation and specificity-aware molecular optimization.**

SurfSpec의 가장 좋은 부분은 “새 diffusion model”이 아닙니다.

핵심은:

$$
\boxed{
\text{target-only geometric mismatch}
\rightarrow
\text{metric structure}
\rightarrow
\text{separated off-target bound}
\rightarrow
\text{specificity-aware optimization principle}
}
$$

라는 연결입니다.

실험도 의미가 있습니다. 단순 size extension이 mismatch를 더 낮추면서도 clash와 empirical specificity를 악화시키는 결과는 **proxy metric을 그대로 maximize하면 안 된다**는 좋은 negative control입니다.

반면 claim은 좁게 유지해야 합니다. Main evaluation은 CrossDocked2020의 random off-target pockets와 AutoDock Vina proxy에 의존하고, theorem도 geometric separation과 empirical geometry–affinity calibration을 요구합니다. 따라서 experimental selectivity나 open-world safety로 확장하면 안 됩니다.

이 논문을 실제 연구에 가져올 때 가장 먼저 재사용할 것은 generator가 아니라 **specificity-aware evaluation contract**입니다.

## 6개월 뒤 기억해야 할 세 가지

1. **Affinity와 specificity는 다른 objective다.** Target Vina 하나가 좋아져도 strongest off-target margin은 나빠질 수 있다.
2. **SurfSpec의 진짜 novelty는 surface mismatch를 metric으로 만들어 triangle inequality 기반 certificate로 연결한 것**이다. 다만 guarantee는 geometrically separated off-target와 empirical calibration assumption 안에서만 성립한다.
3. **Mismatch만 낮추면 안 된다.** Naive size extension은 더 낮은 mismatch를 만들면서도 clash와 empirical specificity를 악화시킨다. Controlled growth + molecular prior + hard-negative evaluation이 같이 필요하다.

## Related

- [[papers/sbdd/index|Structure-Based Modeling Papers]]
- [[molecular-modeling/structure-based/index|Structure-Based Modeling]]
- [[concepts/sbdd/binding-affinity|Binding Affinity]]
- [[concepts/sbdd/pose-quality|Pose Quality]]
- [[concepts/sbdd/virtual-screening|Virtual Screening]]
- [[concepts/evaluation/negative-set|Negative Set]]
- [[concepts/evaluation/applicability-domain|Applicability Domain]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[papers/generative-models/lift|LiFT]]

## Sources

- [arXiv:2609.02963](https://arxiv.org/abs/2609.02963)
- [SurfSpec PDF](https://arxiv.org/pdf/2609.02963)
- Francoeur et al., CrossDocked2020
- Schneuing et al., DiffSBDD
- Igashov et al., DiffLinker
