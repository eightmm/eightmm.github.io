---
title: Controllable Molecular Generation with Fine-Tuned Flow Matching
aliases:
  - papers/flow-matching-rl
  - papers/generative-models/flow-matching-rl
tags:
  - papers
  - generative-models
  - flow-matching
  - reinforcement-learning
  - molecular-generation
  - conditional-generation
  - structure-based-modeling
status: full-note
source_type: Journal
source_url: https://doi.org/10.1038/s42004-026-02188-z
---

# Controllable Molecular Generation with Fine-Tuned Flow Matching

> **한 줄 요약:** 이미 학습된 3D Flow Matching generator를 다시 처음부터 설계하지 않고, 임의의 black-box reward를 **Flow Matching training loss의 sample weight**로 바꿔 continuous coordinates와 discrete chemistry를 함께 post-train하는 방법이다. 가장 중요한 교훈은 reward steering 자체보다도, **reward hacking과 prior drift를 어떻게 감시하고 제어할 것인가**에 있다.

## 왜 이 논문을 저장하는가

이 논문은 “더 좋은 base generator”를 만드는 논문이라기보다 **base generator 이후의 optimization layer**를 정의한다.

실제 SBDD에서는 목표가 자주 바뀐다.

- 한 프로젝트에서는 permeability proxy를 개선하고 싶다.
- 다른 프로젝트에서는 target activity predictor를 올리고 싶다.
- pocket-conditioned generation에서는 pose complementarity나 docking-like proxy를 개선하고 싶다.
- 어느 순간에는 여러 목표를 동시에 만족시켜야 한다.

모든 목표를 base model의 condition으로 미리 넣을 수는 없다. 반대로 매번 새 generator를 학습하는 것도 비싸다. 이 논문은 이 지점을 다음 문제로 바꾼다.

$$
\text{pretrained generator}
+
\text{project-specific reward}
\rightarrow
\text{post-trained generator}.
$$

핵심은 reward가 differentiable할 필요가 없다는 점이다. 생성된 molecule을 외부 evaluator에 넣어 scalar reward만 받을 수 있으면 된다. 따라서 docking score, chemistry filter, learned predictor, force-field score처럼 gradient를 generator까지 직접 전달하기 어려운 objective도 동일한 interface로 취급할 수 있다.

이 설계는 3D molecular generation에서 특히 유용하다. molecule state가 단순한 coordinate vector가 아니라

$$
z = (x, a, b, c)
$$

처럼

- $x$: atom coordinates,
- $a$: atom types,
- $b$: bond types,
- $c$: formal charges

를 동시에 포함하기 때문이다. 이 논문의 post-training은 좌표만 움직이는 guidance가 아니라 **continuous/discrete flow 전체를 다시 기울인다.**

---

## Figure guide

### Figure 1 — reward가 분포를 실제로 어떻게 이동시키는가

![Paper Figure 1 — PSA optimization](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs42004-026-02188-z/MediaObjects/42004_2026_2188_Fig1_HTML.png)

*Source: Wang, Janet & Tibo, Communications Chemistry 9, 291 (2026), Fig. 1. CC BY 4.0. 이 그림은 저자 보고 결과다. (a)는 molecule size가 달라도 PSA distribution이 agent 쪽에서 낮아지는지, (b)는 RL epoch에 따라 batch-median PSA가 내려가는지, (c)는 같은 initial noise/size에서 prior와 agent가 어떤 chemical change를 만드는지를 보여준다. 중요한 점은 “reward가 좋아졌다”가 아니라 **generator distribution 자체가 reward-favored region으로 이동한다**는 것이다.*

### Figure 4 — pocket reward hacking이 왜 핵심 문제인가

![Paper Figure 4 — pocket-conditioned interaction-energy optimization](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs42004-026-02188-z/MediaObjects/42004_2026_2188_Fig4_HTML.png)

*Source: Wang, Janet & Tibo, Communications Chemistry 9, 291 (2026), Fig. 4. CC BY 4.0. 이 그림 역시 저자 보고 결과다. UCK2 pocket에서 agent가 더 favorable한 interaction-energy 쪽으로 분포를 이동시키고 representative pose를 바꾸는 것을 보여준다. 그러나 이 결과를 binding affinity 개선으로 읽으면 안 된다. 저자들도 사용한 MMFF94 interaction term이 binding energy나 free energy가 아니라고 명시한다.*

> **Figure reading rule:** 두 그림을 함께 보면 논문의 장점과 위험이 동시에 보인다. Fig. 1은 reward-weighted post-training이 distribution steering에 실제로 작동함을 보여주고, Fig. 4는 그 steering이 physical objective를 잘못 정의하면 shortcut을 찾을 수 있음을 보여준다.

---

## Metadata

| Field | Value |
| --- | --- |
| Paper | Controllable molecular generation with fine-tuned flow-matching model |
| Authors | Kunyu Wang, Jon Paul Janet, Alessandro Tibo |
| Venue | Communications Chemistry 9, Article 291 |
| Published | 2026-09-05 |
| DOI | https://doi.org/10.1038/s42004-026-02188-z |
| Official code | https://github.com/MolecularAI/flow_matching_rl |
| Code snapshot inspected | `59617a8c1cbb25f6d50660c5c47213cd66250a98` |
| Base generator | SemlaFlow family |
| Main setting | unconditional + protein-pocket-conditioned 3D molecular generation |
| Public artifacts | code, configs, pretrained checkpoints |
| Article license | CC BY 4.0 |

---

## 1. Problem: conditional generation만으로는 project-specific optimization을 다 담기 어렵다

Conditional generation은 보통 condition $c$를 base training에서 이미 정의한다.

$$
p_\theta(z \mid c).
$$

예를 들어

- target pocket,
- desired property,
- scaffold,
- text condition

을 모델 입력으로 넣을 수 있다.

하지만 실제 discovery에서는 objective가 사후에 바뀐다. Base model이 이미 학습된 뒤에 새로운 property predictor가 생길 수도 있고, 프로젝트마다 bespoke filter가 다를 수도 있다.

이때 선택지는 대략 세 가지다.

1. base model을 새 condition으로 다시 학습한다.
2. inference-time guidance를 건다.
3. pretrained generator 자체를 post-train한다.

이 논문은 3번을 택한다.

그 이유는 practical하다.

- reward가 non-differentiable이어도 된다.
- sampling-time에 매 step evaluator gradient를 계산할 필요가 없다.
- post-trained generator는 이후 sampling에서 이미 objective 쪽으로 distribution이 이동해 있다.
- continuous coordinate와 discrete atom/bond/charge를 함께 바꿀 수 있다.

즉 reward model을 “sampling accessory”가 아니라 **distribution-shaping training signal**로 사용한다.

---

## 2. Minimum background: Flow Matching에서 무엇을 다시 학습하는가

Flow Matching은 simple source distribution에서 data distribution으로 이어지는 probability path를 만들고 그 path의 velocity field를 학습한다.

$$
\frac{d}{dt}\psi_t(x)
=
u_t(\psi_t(x)).
$$

Model은 보통 tractable conditional path를 이용해 velocity 또는 terminal state prediction objective를 학습한다.

Molecular Flow Matching에서는 state가 좌표만이 아니다. 이 논문이 사용하는 SemlaFlow-style state는 mixed continuous/discrete object다.

$$
z_t = (x_t, a_t, b_t, c_t).
$$

학습 loss는 개념적으로

$$
\mathcal{L}_{FM}
=
\lambda_x \mathcal{L}_{coord}
+
\lambda_a \mathcal{L}_{atom}
+
\lambda_b \mathcal{L}_{bond}
+
\lambda_c \mathcal{L}_{charge}
$$

이고 paper/released implementation에서는 coordinate, atom-type, bond, charge term을 함께 사용한다. 공개 코드에서 실제 조합은

$$
\mathcal{L}
=
\mathcal{L}_{coord}
+
0.2\mathcal{L}_{atom}
+
\mathcal{L}_{bond}
+
\mathcal{L}_{charge}.
$$

따라서 reward-weighted post-training은 단순 coordinate refinement가 아니다. Reward가 높았던 final molecule을 설명하는 방향으로 **모든 output channel의 training update를 재가중**한다.

---

## 3. Core idea: reward를 target distribution의 density tilt로 보기

논문의 출발점은 다음 intuition이다.

$$
q_{\text{agent}}(z)
\propto
q_{\text{prior}}(z)\,r(z),
$$

where

- $q_{\text{prior}}$: pretrained generator의 sample distribution,
- $r(z)\in[0,1]$: molecule-level reward,
- $q_{\text{agent}}$: post-training 후 원하는 distribution.

Reward가 큰 region은 더 많은 probability mass를 얻고, reward가 작은 region은 줄어들기를 원한다.

이를 위해 별도의 differentiable reward gradient를 계산하지 않고, generated sample의 Flow Matching loss 자체를 reward로 weighting한다.

가장 단순한 형태는

$$
\mathcal{L}_{RWFM}
=
\mathbb{E}_{z\sim q_{\text{agent}}}
\left[
r(z)\,\mathcal{L}_{FM}(z)
\right].
$$

직관적으로는 high-reward molecule이 training example로 더 강하게 작용한다.

이것은 중요하다. Reward는 “어느 방향으로 coordinate를 움직여라”를 직접 말하지 않는다. 대신

> **이런 terminal molecule을 더 잘 재생산하도록 generator field를 바꿔라**

라고 말한다.

그래서 reward가 black-box여도 된다.

---

## 4. Released code에서 중요한 차이: raw reward가 아니라 centered advantage를 쓴다

공개 구현을 읽으면 실제 update는 reward를 그대로 곱하지 않는다.

```text
advantages = rewards - rewards.mean()
rl_loss = sum(per_sample_loss * advantages)
```

즉

$$
A_i = r_i - \bar r
$$

를 사용하고

$$
\mathcal{L}_{RL}
=
\sum_i A_i\,\mathcal{L}_{FM}^{(i)}
$$

로 update한다.

이 설계는 batch 안에서

- 평균보다 좋은 sample은 한 방향,
- 평균보다 나쁜 sample은 반대 방향

의 상대 signal을 제공한다.

Paper Methods도 batch mean reward를 빼는 것이 convergence에 도움이 된다고 설명한다. 따라서 Eq.의 단순 reward-weighting만 읽고 구현하면 released code와 미묘하게 달라질 수 있다.

이 차이는 reproduction에서 반드시 기록해야 한다.

### 왜 이것이 중요한가

Raw positive reward만 쓰면 모든 sample이 정도 차이는 있어도 같은 부호의 update를 준다.

Centered reward는 batch-relative preference를 만든다.

$$
r_i > \bar r \Rightarrow A_i > 0,
$$

$$
r_i < \bar r \Rightarrow A_i < 0.
$$

따라서 이 방법은 이름은 RL이지만 conventional policy-gradient trajectory estimator보다 **online reward-reweighted generative fine-tuning**에 더 가까운 측면이 있다.

이 distinction을 유지해야 다른 RL molecular-design method와 비교가 명확해진다.

---

## 5. Prior regularization: reward collapse를 막는 최소 안전장치

Online optimization에서 가장 쉬운 실패는 agent가 reward를 과도하게 exploit하면서 prior distribution을 잃는 것이다.

논문은 parameter-space regularization을 둔다.

$$
\Omega(\theta_{\text{agent}})
=
\lambda_r
\left\|
\theta_{\text{agent}}-\theta_{\text{prior}}
\right\|_2^2.
$$

최종 loss는 개념적으로

$$
\mathcal{L}
=
\mathcal{L}_{RL}+\Omega.
$$

논문과 released config에서

$$
\lambda_r=0.1
$$

을 사용한다.

이 regularizer는 다음 contract를 만든다.

> reward를 따라가되 pretrained chemistry prior에서 너무 멀리 가지 않는다.

그러나 이것이 validity나 physical realism을 보장하는 것은 아니다.

Parameter distance는

- molecule validity,
- pose strain,
- diversity,
- scaffold collapse,
- pocket clash

와 직접 동일하지 않다.

따라서 $\lambda_r$는 **collapse-control hyperparameter**이지 chemical-safety certificate가 아니다.

---

## 6. Adaptive molecular-size sampler가 생각보다 중요하다

Molecular generation에서 atom count는 nuisance variable이 아니다. Objective 자체가 molecule size와 강하게 결합될 수 있다.

예를 들어 PSA를 줄이는 가장 쉬운 방법 중 하나는 molecule을 작게 만드는 것이다.

그래서 fixed/uniform size distribution만 두면 reward optimization이 size distribution과 충돌하거나, 반대로 size shortcut을 만들 수 있다.

공개 구현은 각 atom count마다 Beta posterior를 유지한다.

$$
p_i \sim \operatorname{Beta}(\alpha_i,\beta_i).
$$

각 epoch에는 posterior sample을 뽑아 가장 높은 atom count를 선택한다.

그리고 해당 size에서 얻은 mean reward $\bar r$에 대해

$$
\alpha_i \leftarrow \alpha_i + \bar r,
$$

$$
\beta_i \leftarrow \beta_i + 1-\bar r.
$$

이것은 Thompson-sampling-like bandit으로 볼 수 있다.

### 장점

Generator와 molecular size distribution을 동시에 objective에 적응시킬 수 있다.

### 위험

Reward가 size와 강하게 correlated되어 있으면 “좋은 chemistry”가 아니라 “reward가 쉬운 size”를 학습할 수 있다.

따라서 결과를 읽을 때 반드시

$$
p(n_{\text{atoms}})
$$

의 변화도 함께 봐야 한다.

좋은 evaluation은 reward improvement만 보고 끝나면 안 된다.

---

## 7. Reward interface: heterogeneous score를 [0,1]로 정규화한다

실제 molecular objective는 scale이 모두 다르다.

- QED: bounded score,
- LogP: 특정 interval이 좋음,
- PSA: 낮을수록 좋음,
- strain: 낮을수록 좋음,
- predicted activity: 높을수록 좋음,
- interaction energy: 더 negative가 좋음.

논문과 code는 sigmoid family로 raw metric을 $(0,1)$ desirability로 바꾼다.

### Higher-is-better

$$
r(s)=\sigma(s).
$$

### Lower-is-better

reverse sigmoid를 쓴다.

### Preferred interval

double sigmoid를 사용해 interval 안을 선호한다.

Multi-objective에서는 transformed reward를 geometric mean으로 합친다.

두 objective라면

$$
r_{\text{multi}}=\sqrt{r_1r_2}.
$$

세 objective라면

$$
r_{\text{multi}}=(r_1r_2r_3)^{1/3}.
$$

Geometric mean의 장점은 한 objective가 거의 0이면 전체 reward가 강하게 떨어진다는 것이다.

즉 단순 arithmetic average보다 “모든 조건을 어느 정도 만족”하도록 압박한다.

하지만 threshold와 sigmoid slope 자체가 hidden objective design이 된다.

따라서 reward specification은 model hyperparameter의 일부로 봐야 한다.

---

## 8. End-to-end training loop

Released implementation 기준으로 한 epoch는 대략 다음과 같다.

```text
1. AtomSampler에서 molecule size 선택
2. 현재 agent로 batch generation
3. RDKit / predictor / force-field evaluator로 reward 계산
4. valid generated molecules를 terminal state z1로 변환
5. source noise z0와 z1 사이 random-time intermediate zt 구성
6. Flow Matching reconstruction loss를 sample별 계산
7. reward - batch_mean_reward 로 advantage 생성
8. advantage-weighted FM loss + prior parameter regularization
9. Adam update + gradient clipping
10. EMA agent update
11. 해당 atom-count Beta posterior update
```

공개 implementation은 Adam을 사용하고 default learning rate는

$$
10^{-5}
$$

이며 gradient norm은 1.0으로 clip한다.

이 구조의 계산적 장점은 reward derivative가 필요 없다는 것이다.

Evaluator는 완전히 external black box여도 된다.

---

## 9. Training contract: 논문과 release를 함께 읽어야 한다

Paper Methods 기준:

- unconditional prior: SemlaFlow, 약 40M parameters,
- prior data: QM9 + GEOM,
- optimizer: Adam,
- learning rate: $10^{-5}$,
- gradient clipping: 1.0,
- unconditional RL batch size: 64,
- conditional RL batch size: 16,
- reward regularization weight: 0.1,
- 여러 experiment에서 5 independent runs/agents를 사용.

Official repository의 현재 `rl_fastrl.yml`은 pocket-conditioned UCK2 setup을 포함한다.

- `reward: complex_eng_strain`
- `n_epochs: 300`
- `n_min_atoms: 25`
- `n_max_atoms: 45`
- `regular_weight: 0.1`
- `use_protein: True`
- protein: processed PDB 6N53
- native ligand: 6N53 ligand

### Reproduction note: batch-size discrepancy

현재 공개 YAML은 conditional setup에서도 `batch_size: 64`를 적고 있지만 paper Methods는 conditional experiments의 batch size를 16이라고 적는다.

이것은 중요한 작은 discrepancy다.

재현할 때는 단순히 current YAML을 “paper exact config”로 가정하지 말고,

1. paper version,
2. repository commit,
3. config file,
4. checkpoint metadata

를 함께 pin해야 한다.

---

## 10. Evaluation contract: 무엇이 independent evidence이고 무엇이 reward 자체인가

이 논문에서 가장 중요한 reading rule은 다음이다.

$$
\text{optimized reward}
\neq
\text{independent validation}.
$$

일부 experiment는 training reward와 evaluation metric이 사실상 같은 evaluator family를 공유한다.

예를 들어 DRD2 task에서 activity predictor를 reward로 쓰고 그 predicted activity를 다시 주요 결과로 보면, 이것은 **predictor optimization evidence**이지 biological activity evidence가 아니다.

따라서 metric을 세 층으로 나눠야 한다.

### Layer A — optimized objective

- PSA
- predicted DRD2 activity
- LogP
- MMFF94 interaction energy
- strain when reward에 포함

### Layer B — internal quality diagnostics

- validity
- uniqueness
- QED
- strain
- molecular size distribution

### Layer C — 더 독립적인 downstream checks

- independent docking score
- PoseBusters-like geometry validity
- alternative force field / energy evaluator
- unseen pocket performance
- experimental assay

논문의 strongest evidence는 A와 B다.

C는 제한적이며 prospective wet-lab evidence는 없다.

---

## 11. Experiment 1: PSA optimization은 method가 작동한다는 가장 깨끗한 sanity check다

Agent는 low PSA를 reward로 학습한다.

5000 generated samples 비교에서 저자 보고 평균은:

| Metric | Prior | Agent |
| --- | ---: | ---: |
| PSA ↓ | 67.51 Å² | **9.00 Å²** |
| Validity ↑ | 85.8% | **89.8%** |
| QED ↑ | 0.54 | 0.54 |
| Strain ↓ | **2.27** | 3.85 kcal/mol/atom |

이 결과는 매우 명확하다.

### 무엇을 지지하는가

Reward-weighted post-training이 생성 distribution을 원하는 physicochemical region으로 강하게 이동시킬 수 있다.

### 무엇을 동시에 경고하는가

Optimized metric 하나가 크게 좋아져도 orthogonal physical metric은 나빠질 수 있다.

PSA는 극적으로 낮아졌지만 strain은 증가한다.

따라서

$$
\text{reward improvement}
\not\Rightarrow
\text{global molecule quality improvement}.
$$

이것이 이후 pocket experiment에서 더 심각한 형태로 반복된다.

---

## 12. Experiment 2: learned activity predictor를 reward로 쓸 수 있다

DRD2 example에서는 external activity predictor를 reward source로 사용한다.

저자 보고 기준:

- predicted $p(\text{active})$: 0.37 → 0.41,
- validity: 84.1% → 90.8%,
- QED: 0.54 → 0.59,
- uniqueness: 100% 유지,
- strain은 악화.

이것은 method의 black-box compatibility를 보여준다.

Reward가 analytic property일 필요가 없다.

### 하지만 claim boundary가 중요하다

이 결과는

> agent가 DRD2 experimental binder를 더 많이 만든다

를 증명하지 않는다.

정확한 claim은

> agent가 **사용한 DRD2 predictor가 높은 score를 주는 molecule distribution**으로 이동했다

이다.

Predictor exploitation 가능성은 항상 남는다.

그래서 practical pipeline에는

- training reward predictor와 다른 model,
- structure-based evaluator,
- applicability-domain check,
- novelty/scaffold diagnostics

가 필요하다.

---

## 13. Experiment 3: multi-objective optimization에서 geometric mean이 실제로 역할을 한다

LogP와 predicted activity를 동시에 optimize하는 experiment는 단일-objective reward의 한계를 보여준다.

한쪽만 optimize하면 다른 축을 반드시 만족하지 않는다.

Dual reward를 사용하면 두 조건을 동시에 만족하는 molecule 수가 증가한다.

저자 보고 기준:

- prior: 5000개 중 176개가 $p>0.5$ 및 LogP 0.5–3.5를 동시에 만족.
- dual-objective agent: **346/5000**.

거의 두 배다.

이 결과는 post-training interface의 가장 practical한 장점이다.

새로운 objective combination이 생길 때 base generator architecture를 다시 설계하지 않고 reward composition만 바꿀 수 있다.

그러나 objective count가 늘수록

- reward scaling,
- threshold,
- incompatibility,
- Pareto diversity

문제가 커진다.

하나의 scalar geometric mean만으로 medicinal chemistry trade-off 전체를 표현할 수 있다는 뜻은 아니다.

---

## 14. Experiment 4: pocket-conditioned generation이 이 논문의 가장 중요한 failure-analysis case다

Conditional experiment는 UCK2, PDB **6N53**를 사용한다.

목표는 pocket 안에서 ligand–protein intermolecular interaction energy를 더 favorable하게 만드는 것이다.

Reward source는 MMFF94 기반 electrostatic + van der Waals interaction term이다.

저자들은 매우 중요한 경계를 직접 명시한다.

이 값은

- bound vs unbound state difference가 아니므로 binding energy가 아니고,
- entropy를 포함하지 않으므로 free energy도 아니다.

즉

$$
\Delta E_{\text{interaction}}
\neq
\Delta G_{\text{bind}}.
$$

### 첫 번째 실패

Interaction energy만 줄이면 agent가 **high-strain ligand pose**를 만들기 시작한다.

이것은 textbook reward hacking이다.

Model 입장에서는 reward가 요구한 것을 정확히 수행했다.

문제는 objective가 incomplete했다는 것이다.

### 수정

Reward를 interaction + strain dual objective로 바꾼다.

$$
r
=
\sqrt{r_{\text{interaction}}r_{\text{strain}}}.
$$

이후 저자 보고에서는 agent가 more favorable interaction-energy distribution을 만들면서 pathological strain을 줄이는 방향으로 이동한다.

이 사례가 논문에서 가장 오래 기억할 부분이다.

> **SBDD reward는 evaluator가 아니라 specification이다. 빠진 physical constraint는 model이 exploit할 수 있는 loophole다.**

---

## 15. Pocket condition 자체의 boundary

Conditional generator는 protein coordinates를 사용하고, native ligand 정보가 pocket setup에 들어간다.

Paper Methods에서는 pocket을 native ligand 주변 residue로 구성한다.

따라서 이 experiment는

- known bound pocket,
- known receptor structure,
- specific target 6N53

조건에서의 post-training이다.

이 결과를 다음으로 확장하면 안 된다.

- global binding-site discovery,
- apo pocket recognition,
- broad target-family OOD,
- induced-fit modeling,
- prospective affinity optimization.

특히 protein preparation, protonation, pocket definition, native-ligand context는 deployment contract의 일부다.

---

## 16. What is actually novel?

### 16.1 Reward-weighted generation 자체

Reward-weighted generative fine-tuning은 완전히 새로운 일반 개념은 아니다.

### 16.2 Flow Matching에 reward를 넣는 것

Flow/diffusion post-training도 이미 넓은 연구 축이 있다.

### 16.3 이 논문의 실질적 novelty

Molecular setting에서 가장 중요한 contribution은 다음 조합이다.

$$
\boxed{
\text{mixed continuous/discrete 3D FM}
+
\text{black-box reward}
+
\text{online post-training}
+
\text{adaptive size}
+
\text{prior regularization}
}
$$

즉 3D molecule의

- coordinate,
- atom type,
- bond type,
- charge,
- molecule size

까지 함께 project-specific objective 쪽으로 이동시키는 end-to-end recipe다.

이 조합이 실제 code와 checkpoint로 공개되어 있다는 점도 중요하다.

---

## 17. Why not just use inference-time guidance?

Inference-time guidance와 post-training은 다른 compute contract를 가진다.

### Guidance

매 generation run에서 evaluator/gradient 또는 auxiliary process가 반복적으로 필요할 수 있다.

### Post-training

한 번 agent를 fine-tune하면 이후 sampling은 agent distribution에서 바로 나온다.

대규모 screening을 반복한다면 amortization이 가능하다.

하지만 post-training의 downside도 있다.

- reward 바뀌면 다시 학습해야 한다.
- distribution collapse가 누적될 수 있다.
- objective-specific agent를 여러 개 관리해야 한다.
- training-time evaluator bias가 weights에 박힌다.

따라서 어느 쪽이 더 좋은지는

$$
\text{number of samples}
\times
\text{evaluator cost}
\times
\text{objective lifetime}
$$

에 따라 달라진다.

---

## 18. Reproducibility audit

공개 repository는 reproduction 관점에서 좋은 편이다.

확인 가능한 항목:

- training entrypoint,
- YAML config,
- reward transforms,
- reward composition,
- atom-size sampler,
- prior regularization,
- pocket-conditioned setup,
- checkpoint loading,
- seed override,
- regularization sweep override.

### Released code에서 직접 확인되는 핵심

#### Prior penalty

```text
sum((theta_agent - theta_prior)^2)
```

#### Centered reward

```text
reward - reward.mean()
```

#### Atom count update

```text
alpha[n] += mean_reward
beta[n] += 1 - mean_reward
```

#### Multi-channel FM loss

```text
coord + 0.2 * atom_type + bond + charge
```

#### Pocket reward

```text
geometric_mean(
    transformed_interaction_energy,
    transformed_strain
)
```

이 정도면 논문 아이디어만 공개된 상태가 아니라 실제 post-training loop를 audit할 수 있다.

---

## 19. Reproduction path

재현한다면 순서는 다음이 좋다.

1. Released pretrained prior checkpoint를 고정한다.
2. Official repo commit을 pin한다.
3. Prior sampling metric을 먼저 재현한다.
4. PSA single-objective를 sanity test로 돌린다.
5. Reward-centering on/off를 비교한다.
6. $\lambda_r$ sweep으로 collapse frontier를 그린다.
7. AtomSampler를 fixed/uniform/adaptive로 분리한다.
8. Multi-objective reward를 재현한다.
9. 마지막에 pocket-conditioned 6N53를 돌린다.
10. 동일 generated set에 independent metrics를 추가한다.

### 반드시 저장할 것

- code commit,
- base checkpoint hash,
- reward definition,
- sigmoid thresholds/slopes,
- batch size,
- atom-count range,
- seed,
- stopping epoch,
- number of generated candidates,
- invalid-sample policy,
- final evaluator version.

---

## 20. 가장 중요한 ablation: 무엇이 gain을 만드는가

이 논문을 architecture/objective component로 나누면 최소 다음 실험이 필요하다.

| Arm | Reward weighting | Centering | Prior reg | Adaptive size |
| --- | --- | --- | --- | --- |
| A | no | — | — | no |
| B | yes | no | no | no |
| C | yes | yes | no | no |
| D | yes | yes | yes | no |
| E | yes | yes | yes | yes |

이렇게 해야 다음을 분리할 수 있다.

- reward weighting 자체,
- centered advantage 효과,
- regularization 효과,
- atom-size adaptation 효과.

현재 full method 성능만 보면 이 네 요소의 marginal value를 완전히 분리하기 어렵다.

---

## 21. SBDD에 적용할 때 필요한 evaluation panel

Pocket-conditioned reward를 사용할 때 primary reward 하나로 판단하면 안 된다.

### Optimized metrics

- chosen docking/interaction reward,
- reward success rate.

### Geometry checks

- PoseBusters,
- intramolecular strain,
- protein–ligand clashes,
- bond/angle outliers,
- chirality/stereo validity.

### Chemistry checks

- validity,
- QED,
- SA,
- LogP,
- charge distribution,
- ring statistics.

### Distribution checks

- uniqueness,
- scaffold diversity,
- novelty,
- atom-count distribution,
- nearest-neighbor similarity to prior/train.

### Independent target checks

- second docking/scoring function,
- rescoring after minimization,
- alternative protein conformation,
- target-family OOD,
- experimental assay where possible.

이 panel을 통과해야 “reward optimization”을 “useful SBDD optimization”으로 승격할 수 있다.

---

## 22. Generalization test: one-pocket result에서 벗어나기

6N53 하나에서 잘 되는 것과 transferable pocket steering은 다른 claim이다.

추천 split은 다음과 같다.

### Target OOD

Protein sequence/structure family cluster 단위 split.

### Ligand OOD

Bemis–Murcko scaffold split.

### Joint OOD

$$
\text{unseen protein family}
+
\text{unseen ligand scaffold}.
$$

### Reward OOD

Training reward와 다른 independent evaluator에서 유지되는지 확인.

### Conformation OOD

같은 target의 alternative receptor structure/apo structure에서 유지되는지 확인.

이렇게 해야 agent가

- specific protein geometry,
- specific scoring function,
- specific ligand-size range

를 외운 것인지 분리할 수 있다.

---

## 23. Reward hacking을 실험적으로 어떻게 측정할 것인가

Reward hacking은 qualitative anecdote로만 보면 안 된다.

다음 곡선을 그릴 수 있다.

$$
\text{optimized reward}
\quad \text{vs} \quad
\text{independent physical validity}.
$$

Training epoch마다

- reward,
- strain,
- clash count,
- PoseBusters pass,
- QED,
- diversity,
- atom-count entropy

를 같이 기록한다.

만약 reward가 계속 좋아지는데 independent metric이 특정 epoch 이후 악화된다면 early stopping criterion을 만들 수 있다.

즉 stopping rule을

$$
\arg\max_t r_t
$$

가 아니라

$$
\arg\max_t U(
r_t,
\text{validity}_t,
\text{diversity}_t,
\text{physics}_t
)
$$

형태로 바꾸는 것이 더 안전하다.

---

## 24. 내가 가장 먼저 해볼 실험

하나의 pocket-conditioned pretrained 3D FM을 고정한다.

### Variants

1. prior only
2. reward-weighted, no regularization
3. reward-weighted + prior regularization
4. reward-weighted + reg + adaptive size
5. inference-time guidance baseline

### Rewards

- docking only
- strain only
- docking + strain
- docking + strain + QED/SA constraint
- learned affinity/scoring model + physics constraint

### 동일 budget

모든 방법에서

- same evaluator calls,
- same number of generated candidates,
- same downstream filtering budget

을 맞춘다.

### Decision criterion

방법이 의미 있으려면 reward 하나가 아니라

$$
\text{enrichment}
+
\text{validity}
+
\text{diversity}
+
\text{OOD robustness}
$$

의 Pareto frontier를 개선해야 한다.

---

## 25. Failure modes

| Failure mode | 왜 위험한가 |
| --- | --- |
| reward predictor exploitation | predicted score만 좋아지고 실제 property는 개선되지 않을 수 있음 |
| strain shortcut | intermolecular reward를 위해 internally distorted ligand를 만들 수 있음 |
| atom-size shortcut | objective가 쉬운 molecule size로 distribution이 collapse할 수 있음 |
| scaffold collapse | 높은 reward scaffold 몇 개에 probability mass가 집중될 수 있음 |
| evaluator overfit | 한 docking/ML model의 bias가 agent weights에 고정됨 |
| single-pocket overfit | target-specific geometry를 transferable rule로 오해할 수 있음 |
| invalid-sample filtering bias | invalid output 제외 후 metric만 보고하면 개선이 과장될 수 있음 |
| reward scaling sensitivity | sigmoid threshold/slope가 사실상 optimization objective를 바꿈 |
| early stopping on reward only | independent physical metrics가 악화된 뒤에도 training이 계속될 수 있음 |
| current-config drift | paper config와 repository current config가 달라 exact reproduction이 깨질 수 있음 |

---

## 26. 이 논문이 잘 보여주는 것

### 26.1 Black-box objective도 3D Flow Matching을 post-train할 수 있다

Reward derivative 없이도 generator distribution을 의미 있게 이동시킨다.

### 26.2 Continuous/discrete chemistry를 함께 바꿀 수 있다

Coordinate-only optimization이 아니다.

### 26.3 Adaptive size가 optimization degree of freedom이 될 수 있다

Molecule size를 fixed nuisance variable로 두지 않는다.

### 26.4 Multi-objective reward가 single-objective shortcut을 줄일 수 있다

Pocket example에서 특히 명확하다.

### 26.5 Reward hacking이 실제로 발생한다

이것이 오히려 논문의 가치다. Failure가 숨겨지지 않고 method design의 일부가 된다.

---

## 27. 이 논문이 증명하지 못한 것

### 27.1 Experimental affinity improvement

없다.

### 27.2 Broad pocket generalization

Pocket example은 specific UCK2 context다.

### 27.3 True binding free-energy optimization

MMFF94 interaction energy는 $\Delta G_{\text{bind}}$가 아니다.

### 27.4 Reward model robustness

DRD2 predictor score improvement가 independent biological evidence는 아니다.

### 27.5 Universal superiority over inference-time guidance

Matched compute/evaluator-budget comparison이 더 필요하다.

---

## 28. 다른 개념과 연결

### [[concepts/generative-models/flow-matching|Flow Matching]]

Base training objective와 post-training update를 이해하는 핵심.

### [[concepts/generative-models/conditional-generation|Conditional generation]]

Base condition과 post-hoc reward steering의 역할 차이가 중요하다.

### [[concepts/generative-models/molecular-generation|Molecular generation]]

Validity/diversity/novelty/task utility를 분리해서 읽어야 한다.

### [[molecular-modeling/structure-based/index|Structure-based modeling]]

Pocket preparation, pose validity, scoring proxy와 experimental affinity의 boundary를 유지해야 한다.

### [[papers/generative-models/lift|LiFT]]

LiFT가 external semantic prior를 state-dependent routing으로 넣는다면, 이 논문은 **generator weights 자체를 reward에 맞게 post-train**한다. 둘은 controllability의 서로 다른 축이다.

### [[papers/sbdd/neat-pocket|NEAT-POCKET]]

Base generator architecture를 바꾸는 접근과 post-training objective를 바꾸는 접근을 분리해서 비교할 수 있다.

---

## 29. Final verdict

**Verdict: Must Read for controllable 3D molecular generation.**

가장 중요한 contribution은 “RL을 썼다”가 아니다.

기억해야 할 것은 다음 architecture-independent recipe다.

$$
\boxed{
\text{pretrained generative prior}
+
\text{black-box project reward}
+
\text{reward-reweighted self-training}
+
\text{prior preservation}
}
$$

이 recipe는 SBDD workflow와 잘 맞는다. Base generator를 한 번 크게 학습하고 프로젝트마다 reward layer만 바꾸는 방식이 가능하기 때문이다.

하지만 이 논문은 동시에 아주 중요한 경고를 준다.

> **Reward는 평가 지표가 아니라 model specification이다. 빠진 constraint는 optimization loophole가 된다.**

따라서 실제 drug-discovery application에서 핵심은 reward를 더 강하게 만드는 것이 아니라, reward와 독립적인 physical/chemical/OOD evaluation panel을 함께 설계하는 것이다.

---

## 6개월 뒤 기억해야 할 세 가지

1. **이 방법은 inference guidance가 아니라 generator post-training이다.** Reward가 Flow Matching loss의 sample weighting을 바꾸면서 coordinates와 discrete chemistry를 함께 재학습한다.
2. **공개 code는 centered reward와 prior L2 regularization, adaptive atom-count Beta sampler를 사용한다.** Paper의 conceptual equation만 보고 구현하면 중요한 detail을 놓칠 수 있다.
3. **UCK2 experiment의 핵심 결과는 binding improvement가 아니라 reward hacking의 발견이다.** Interaction energy만 optimize하자 strain shortcut이 생겼고, multi-objective reward가 필요해졌다.

---

## Sources

- Wang, K., Janet, J. P. & Tibo, A. **Controllable molecular generation with fine-tuned flow-matching model.** *Communications Chemistry* 9, 291 (2026). https://doi.org/10.1038/s42004-026-02188-z
- Official implementation: https://github.com/MolecularAI/flow_matching_rl
- Official archived code artifact / citation: https://doi.org/10.5281/zenodo.21806881
- Released checkpoints referenced by Research OS: https://zenodo.org/records/21790076
- SemlaFlow: https://doi.org/10.48550/arXiv.2406.07266

## Figure license

Figures 1 and 4 are reproduced from the open-access article under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Source and figure identity are stated directly above; no scientific content was modified.
