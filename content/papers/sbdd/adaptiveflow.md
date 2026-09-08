---
title: AdaptiveFlow — AI-enhanced adaptive virtual screening of large libraries for ligand discovery
aliases:
  - papers/adaptiveflow
  - papers/sbdd/adaptiveflow
tags:
  - papers
  - sbdd
  - structure-based-modeling
  - virtual-screening
  - docking
  - active-learning
  - chemical-space
  - hpc
status: full-note
source_type: Journal
source_url: https://www.nature.com/articles/s41587-026-03217-x
---

# AdaptiveFlow: AI-enhanced adaptive virtual screening of large libraries for ligand discovery

> **한 줄 요약:** AdaptiveFlow의 핵심은 69-billion-compound library를 전부 dock하는 것이 아니라, **chemical space를 physicochemical tranche grid로 coarse하게 cover하고 target-specific prescreen evidence로 계산 예산을 promising regions에 집중하는 adaptive search policy**를 virtual-screening infrastructure와 결합한 것입니다.

## 왜 이 논문을 저장하는가

Ultra-large virtual screening(ULVS)에서 흔히 모델만 봅니다.

- 어떤 docking engine이 더 정확한가?
- 어떤 learned scoring function이 더 높은 enrichment를 내는가?
- 어떤 GPU implementation이 더 빠른가?

하지만 library가 $10^9$–$10^{11}$ 규모로 커지면 더 근본적인 질문이 생깁니다.

> **어떤 molecule을 아예 평가할 것인가?**

Exhaustive screening의 계산량을 단순화하면

$$
C_{\mathrm{exhaustive}}
\approx
N\,c_{\mathrm{dock}},
$$

where

- $N$: library size,
- $c_{\mathrm{dock}}$: molecule 하나를 평가하는 평균 비용입니다.

$N$이 69 billion이면 docking model을 2배 빠르게 만드는 것보다 **평가해야 하는 후보의 수 자체를 100배 줄이는 search policy**가 더 큰 leverage를 가질 수 있습니다.

AdaptiveFlow는 이 관점에서 중요합니다. 논문의 contribution은 새로운 neural docking architecture 하나가 아니라 다음 네 층을 하나의 system으로 묶는 데 있습니다.

1. **search-ready chemical-space representation**
2. **coarse-to-fine target-guided sampling policy**
3. **classical / ML docking을 교체 가능한 execution layer로 만드는 infrastructure**
4. **실험 hit와 crystal structure까지 연결한 prospective validation**

따라서 이 논문은 `virtual screening software`라기보다 **compute-budgeted search under an expensive oracle**의 사례로 읽는 편이 더 오래 남습니다.

---

## Metadata

| Field | Value |
| --- | --- |
| Paper | AI-enhanced adaptive virtual screening of large libraries for ligand discovery |
| Journal | Nature Biotechnology |
| Published | 2026-09-01 |
| DOI | 10.1038/s41587-026-03217-x |
| Main system | AdaptiveFlow |
| Main library | Enamine REAL Space 2022q1-2 |
| Ready-to-dock scale | 68.7 billion prepared molecules |
| Chemical-space index | up to 18-dimensional physicochemical tranche grid |
| Occupied tranches | >12 million |
| Mean molecules / populated tranche | about 5,600 |
| Screening policy | Adaptive Target-Guided Virtual Screening (ATG-VS) |
| Optional learned filter | Morgan fingerprint → MLP classifier |
| Prospective targets | FSP1, PARP1 |
| Source code | AdaptiveFlow-LP, AdaptiveFlow-VS, AdaptiveFlow-Unity |
| Code license | GNU GPL v2.0 |

> **Claim boundary:** 아래 성능 수치와 prospective 결과는 논문 저자들이 보고한 결과입니다. Docking enrichment, biochemical potency, structure validation, cellular utility는 서로 다른 evidence layer이므로 하나로 합쳐 해석하지 않습니다.

---

## Figure guide — 먼저 이 세 장을 보면 된다

이 논문은 architecture보다 **system + search policy**가 핵심이라 figure를 같이 보는 것이 특히 중요합니다. 저작권/재사용 경계가 불명확한 third-party image를 복제하지 않고, 공식 Nature Biotechnology 원문 figure에 직접 연결합니다.

### Figure 1 — AdaptiveFlow 전체 system

[Official paper — Fig. 1: The AdaptiveFlow platform for ULVSs](https://www.nature.com/articles/s41587-026-03217-x#Fig1)

**볼 것:** AFLP(ligand preparation), AFVS(virtual screening), AFU(unified workflow)가 별도 module이면서 하나의 execution stack으로 이어지고, classical docking뿐 아니라 ML/DL docking, CPU/GPU, cloud/HPC를 같은 orchestration layer에서 다루는 구조입니다.

이 figure가 지지하는 주장은 `새 scoring function`이 아니라 **oracle을 교체 가능한 component로 만들고 search policy와 execution infrastructure를 분리했다**는 것입니다.

### Figure 2 — 69B library를 searchable object로 바꾸는 과정

[Official paper — Fig. 2: Organization and preparation of the Enamine REAL Space](https://www.nature.com/articles/s41587-026-03217-x#Fig2)

**볼 것:** raw molecular library를 ready-to-dock 3D formats로 준비하는 것뿐 아니라 28개 molecular properties를 계산하고, 그중 18개를 이용해 multidimensional tranche space를 구성합니다.

중요한 점은 이것이 visualization convenience가 아니라 **ATG-VS의 indexing structure**라는 것입니다. Search policy는 molecule-by-molecule list 위가 아니라 chemically organized cells 위에서 시작합니다.

### Figure 3 — Adaptive Target-Guided Virtual Screening

[Official paper — Fig. 3: Conceptual workflow of ATG-VSs](https://www.nature.com/articles/s41587-026-03217-x#Fig3)

**볼 것:** 각 tranche에서 1–10 representatives만 먼저 dock하고, 그 evidence로 promising tranches를 선택한 뒤 selected tranches를 더 깊게 screen합니다. Optional ML filter는 이 second stage 안에서 추가 compute allocation을 수행합니다.

이 figure가 논문의 핵심입니다.

$$
\boxed{
\text{chemical-space partition}
\rightarrow
\text{cheap target probe}
\rightarrow
\text{select promising regions}
\rightarrow
\text{spend expensive compute there}
}
$$

---

## 1. Problem: docking accuracy만으로 ULVS를 풀 수 없는 이유

Library size가 작을 때 virtual screening은 대략 다음 문제입니다.

$$
\operatorname{rank}_{m\in\mathcal L} s(m,P),
$$

where $s$는 target $P$에 대한 docking/scoring function입니다.

하지만 library가 수십 billion 규모가 되면 실제 system objective는 달라집니다.

$$
\max_{\pi,\,s}
\operatorname{Utility}
\left(
\text{hits discovered};
\text{oracle calls},
\text{wall time},
\text{money}
\right).
$$

여기서 $\pi$는 **어떤 candidates를 어떤 순서와 fidelity로 평가할지 정하는 policy**입니다.

따라서 두 system이 같은 docking model을 사용하더라도:

- A는 69B를 uniform하게 전부 평가하고,
- B는 먼저 representative probes를 평가한 뒤 0.1–1%의 region에 집중한다면,

실제 drug-discovery system behavior는 매우 다릅니다.

AdaptiveFlow가 해결하려는 것은 이 **screening allocation problem**입니다.

---

## 2. AdaptiveFlow는 세 module로 나뉜다

논문은 platform을 크게 세 component로 정의합니다.

### 2.1 AFLP — AdaptiveFlow Ligand Preparation

AFLP는 massive ligand library를 실제 docking input으로 바꿉니다.

주요 역할은:

- stereoisomer enumeration
- tautomer handling
- desalting / neutralization
- protonation-state preparation
- 3D conformer generation
- geometry validation
- multiple docking formats 변환
- molecular properties 계산
- tranche organization

입니다.

논문에서 REAL Space 준비 후 제공되는 3D formats는 PDB, PDBQT, MOL2, SDF이고, representation / analytics용으로 SMILES, SELFIES, Parquet도 제공합니다.

이 구분은 ML 관점에서도 중요합니다.

$$
\text{molecular identity}
\neq
\text{prepared docking state}.
$$

동일한 vendor molecule도 tautomer, protonation, stereo enumeration을 거치면 실제 screen되는 states 수가 달라집니다. 원래 31.5B 수준으로 기술되는 enumerated chemical space가 preparation 이후 약 68.7B ready-to-dock entries가 되는 이유 중 하나입니다.

### 2.2 AFVS — AdaptiveFlow for Virtual Screening

AFVS는 docking execution과 search orchestration을 담당합니다.

논문은 pose prediction / sampling 방법과 scoring function을 조합해 약 **1,500 protocols**를 지원한다고 보고합니다. 여기에는 classical docking뿐 아니라 일부 neural pose-prediction/scoring approaches도 포함됩니다.

중요한 것은 `1,500 methods를 모두 검증했다`가 아닙니다.

> **1,500 protocols를 같은 interface 아래 교체 가능하게 만들었다.**

이것이 infrastructure contribution입니다.

### 2.3 AFU — AdaptiveFlow Unity

AFU는 ligand preparation과 docking을 하나의 user-facing workflow로 연결합니다. 공식 GitHub repository는 이를 AFLP와 AFVS를 합친 streamlined version으로 설명합니다.

ML 연구자 관점에서는 특히 다음 부분이 유용합니다.

```text
molecule / generated molecule
       ↓
ligand preparation
       ↓
docking / rescoring oracle
       ↓
score / pose / filtered candidates
       ↓
training or optimization loop
```

즉 docking system 자체를 generative model이나 RL/optimization pipeline의 **expensive evaluator**로 사용할 수 있습니다.

---

## 3. 진짜 representation은 ligand embedding이 아니라 chemical-space index다

AdaptiveFlow의 가장 흥미로운 representation choice는 neural embedding이 아닙니다.

AFLP는 각 prepared molecule에 대해 28개 physicochemical / cheminformatic properties를 계산하고, 그중 18개를 선택해 multidimensional grid를 만듭니다.

논문이 기술하는 axes에는 다음과 같은 properties가 포함됩니다.

- molecular weight
- logP
- H-bond donors / acceptors
- rotatable bonds
- TPSA
- logS
- aromatic ring count
- molecular refractivity
- formal charge
- positive / negative charge counts
- fraction $sp^3$
- chiral-center count
- halogen / sulfur counts
- stereoisomer count

이 grid에서 cell 하나가 **tranche**입니다.

전체 18D Cartesian grid의 대부분은 비어 있고, 실제 REAL Space는 12 million이 넘는 populated tranches를 차지합니다. populated tranche당 평균 molecule 수는 약 5,600이지만 distribution은 매우 heavy-tailed라 single-molecule cell부터 $10^7$ 이상 molecule을 포함하는 cell까지 존재합니다.

이 representation의 목적은 예쁜 map이 아닙니다.

$$
\mathcal L
\rightarrow
\{\mathcal T_1,\ldots,\mathcal T_M\}
$$

으로 library를 partition해서, molecule-level expensive search를 **region-level decision problem**으로 올리는 것입니다.

---

## 4. ATG-VS: coarse-to-fine search policy

Adaptive Target-Guided Virtual Screening은 다음 구조입니다.

### Stage 1 — prescreen

각 tranche $\mathcal T_j$에서 $k$개의 representatives를 선택합니다.

$$
r_{j,1},\ldots,r_{j,k}\sim\mathcal T_j,
\qquad k\in[1,10]\ \text{in the reported setup}.
$$

이 representatives만 target에 dock합니다.

tranche-level score를 추상적으로

$$
\hat q_j
=
A\big(s(r_{j,1}),\ldots,s(r_{j,k})\big)
$$

처럼 둘 수 있습니다. $A$는 tranche quality를 요약하는 rule입니다.

### Stage 2 — primary screen

prescreen에서 promising한 tranches를 선택합니다.

$$
\mathcal S
=
\operatorname{Select}(\hat q_1,\ldots,\hat q_M;B),
$$

where $B$는 primary-screen budget입니다.

그 다음

$$
\bigcup_{j\in\mathcal S}\mathcal T_j
$$

안의 molecules를 더 깊게 평가합니다.

### Stage 3 — optional higher-fidelity rescoring

상위 candidates는 더 비싼 scoring, receptor flexibility, 다른 docking engine 등으로 다시 평가할 수 있습니다.

즉 AdaptiveFlow가 만드는 것은 단일 ranking이 아니라 **multi-fidelity funnel**입니다.

---

## 5. Compute 관점에서 ATG-VS를 쓰는 이유

단순화하면 ATG-VS 비용은

$$
C_{\mathrm{ATG}}
\approx
M k c_{\mathrm{cheap}}
+
N_{\mathrm{selected}}c_{\mathrm{primary}}
+
N_{\mathrm{final}}c_{\mathrm{expensive}}.
$$

$M$은 occupied tranche 수이고 $k$는 tranche당 representatives 수입니다.

핵심 조건은

$$
N_{\mathrm{selected}} \ll N
$$

이어야 하면서 동시에 hits가 많이 들어 있는 regions를 놓치지 않아야 한다는 것입니다.

이 trade-off는 active screening의 본질입니다.

$$
\text{compute reduction}
\leftrightarrow
\text{false-negative chemical regions}.
$$

논문은 69B REAL Space에서 ATG-VS가 exhaustive search 대비 **up to 1,000× cost reduction**을 제공하면서 strong enrichment를 유지할 수 있다고 보고합니다. 이것은 저자 보고 결과이고 target/protocol dependent claim으로 읽어야 합니다.

---

## 6. Optional ML filtering은 어디에 들어가는가

AdaptiveFlow의 ML layer를 `AI docking model`과 혼동하면 안 됩니다.

ATG-VS에는 prescreen result를 이용해 selected tranches 내부의 molecules를 추가로 거르는 optional classifier가 있습니다.

논문의 reported implementation은:

- Morgan fingerprint
- length 1024
- radius 2
- MLP classifier
- prescreen docking score 상위 quartile을 positive class로 정의

하는 방식입니다.

즉 learned model은 대략

$$
\hat p(y_{\mathrm{good}}=1\mid \operatorname{FP}(m))
$$

을 학습합니다.

저자들은 이 filter가 target에 따라 실제 docking candidates를 대략 30–70% 줄이면서 enrichment를 유지하거나 개선할 수 있다고 보고합니다.

하지만 여기에는 중요한 bias가 있습니다.

> 학습 target 자체가 prescreen docking score이면 classifier는 결국 **docking oracle의 smooth approximation**을 배우는 것입니다.

따라서 이 model이 추가 chemical truth를 학습했다고 해석하면 안 됩니다.

---

## 7. 가장 중요한 failure mode: coarse region selection이 rare chemotype를 버릴 수 있다

ATG-VS의 성공 조건은 locality assumption입니다.

> 비슷한 physicochemical tranche 안에 있는 molecules가 target-specific usefulness에서도 어느 정도 correlated할 것이다.

이 assumption이 맞으면 representative sampling이 효율적입니다.

반대로 activity cliff가 강하거나 useful chemotype이 tranche 내부의 극소수 minority라면 representative가 해당 signal을 보지 못할 수 있습니다.

예를 들어 tranche $\mathcal T$에 10,000 molecules가 있고 truly useful ligand가 3개뿐이라면 random representative 하나가 그 signal을 포착할 확률은 매우 낮습니다.

이 때문에 ATG-VS benchmark는 단순 enrichment 외에도 반드시 다음을 봐야 합니다.

- missed chemotypes
- scaffold diversity of recovered hits
- tranche-level false-negative rate
- activity-cliff recovery
- rare-region recall

AdaptiveFlow 논문은 search efficiency를 강하게 보여주지만 이 축은 향후 더 깊게 검증할 가치가 있습니다.

---

## 8. Representative 수와 docking exhaustiveness는 서로 다른 budget axis다

논문은 prescreen에서 적어도 다음 configurations를 비교합니다.

1. 1 representative / tranche, exhaustiveness 1
2. 10 representatives / tranche, exhaustiveness 1
3. 1 representative / tranche, exhaustiveness 10

이 비교가 좋은 이유는 **breadth와 depth를 분리**하기 때문입니다.

- representatives를 늘리기: 같은 cell 내부 chemical diversity를 더 많이 본다.
- exhaustiveness를 늘리기: 한 molecule의 pose-search depth를 높인다.

즉 동일 compute를 어디에 쓸 것인가의 문제입니다.

$$
\text{more molecules}
\quad\text{vs}\quad
\text{better evaluation per molecule}.
$$

ULVS에서 이 비교는 docking architecture benchmark보다도 실무적으로 중요할 수 있습니다.

---

## 9. Evaluation contract: screening score 하나로 끝내면 안 된다

Adaptive virtual screening을 평가할 때 최소 다섯 axis를 분리해야 합니다.

| Axis | 질문 |
| --- | --- |
| Retrieval | fixed budget에서 active compounds / strong binders를 얼마나 회수하는가? |
| Diversity | 같은 scaffold만 반복해서 고르는가? |
| Cost | docking calls, CPU/GPU hours, money가 얼마인가? |
| Wall time | 실제 orchestration을 포함한 turnaround가 얼마인가? |
| Prospective utility | synthesis / assay 이후 실제 hits가 나오는가? |

여기에 structure-based pipeline이라면 추가로:

- pose plausibility
- docking-score calibration
- receptor preparation sensitivity
- protonation/tautomer sensitivity
- target-family dependence

를 봐야 합니다.

따라서

$$
\text{enrichment improvement}
\not\Rightarrow
\text{experimental hit-rate improvement}
$$

이고

$$
\text{docking score}
\not\Rightarrow
K_d, K_i, IC_{50}.
$$

AdaptiveFlow가 강한 이유는 prospective assay와 structure validation까지 일부 연결했다는 점이지, docking proxy가 biochemical truth가 되었기 때문은 아닙니다.

---

## 10. Test-set benchmark와 69B production screen은 같은 evidence가 아니다

논문은 두 가지 평가 scale을 사용합니다.

### Small / controlled benchmark

빠른 비교를 위해 Enamine space에서 약 5 million molecules 규모 subsets를 구성하고, 여러 target에서:

- random / standard ULVS-like selection
- ATG without active learning
- ATG + optional ML

등을 비교합니다.

이 setting은 **policy ablation**에 좋습니다.

### Full-scale production benchmark

전체 69B library scale에서도 ATG-VS를 실행합니다. 이 scale에서는 active-learning step을 생략한 production benchmarks도 사용합니다.

이 setting은 **scalability / feasibility evidence**에 좋습니다.

둘을 섞으면 안 됩니다.

> 5M controlled benchmark에서 가장 좋은 policy가 69B full-production setting에서도 같은 ranking을 보인다는 것은 별도 주장입니다.

---

## 11. HPC system contract: model speed가 아니라 throughput system을 본다

AdaptiveFlow는 Slurm과 AWS Batch를 포함한 heterogeneous execution을 지원합니다.

논문은 AWS에서 최대 **5.6 million vCPUs** 규모까지 near-linear scaling을 보고합니다. AFLP / screening workload는 embarrassingly parallel component가 크기 때문에 이런 scaling이 가능한 구조입니다.

하지만 이 숫자를 읽을 때 다음을 구분해야 합니다.

### strong evidence

- system이 매우 큰 distributed job graph를 실제로 orchestration할 수 있다.
- collection / subjob abstraction으로 scheduler overhead를 관리한다.
- spot/preemptible capacity를 고려한 execution support가 있다.

### 이 숫자만으로 말할 수 없는 것

- 모든 docking protocol이 동일하게 near-linear scale한다.
- 모든 target / receptor preparation에서 throughput이 같다.
- cost efficiency가 모든 on-premise setup보다 우월하다.

특히 storage와 I/O도 무시할 수 없습니다. 논문이 제공하는 ready-to-dock REAL Space는 한 3D format당 약 50 TB compressed, 네 formats를 합치면 약 200 TB compressed이며 uncompressed scale은 약 2 PB로 기술됩니다.

ULVS는 model FLOPs만의 문제가 아니라 **data movement + scheduler + failure recovery + storage layout**의 문제입니다.

---

## 12. FSP1 prospective screen: 중요한 것은 pipeline이 assay까지 이어졌다는 점

FSP1 experiment에서 저자들은 coenzyme Q site를 target으로 사용합니다. receptor는 FAD와 NAD+가 결합된 high-resolution structure(PDB 9IFT)를 기반으로 준비했습니다.

Reported workflow의 핵심 숫자는:

- **1 representative per tranche**
- 약 **12 million prescreen dockings**
- prescreen 이후 **10 million molecules** primary screen
- 이 production screen에서는 optional ML feature를 사용하지 않음

입니다.

상위 candidates에는 physicochemical / problematic-group / predicted-toxicity filtering이 추가됩니다.

Notion Research OS에 기록된 저자 보고 결과 기준으로 초기 FSP1 inhibitors 중 두 molecule은 각각 약 **0.283 µM / 0.777 µM $K_i$**를 보였고, structural follow-up에서 co-crystal evidence가 docking-derived binding hypothesis를 지지합니다.

여기서 중요한 점은:

$$
\text{screen score}
\rightarrow
\text{synthesis}
\rightarrow
\text{biochemical assay}
\rightarrow
\text{structure}
$$

로 evidence ladder가 올라간다는 것입니다.

---

## 13. PARP1 validation은 더 명확한 prospective endpoint를 준다

PARP1에서는 ATG prescreen 이후 **100 million molecule primary screen**을 수행하고, 160 candidates를 합성했다고 보고합니다.

그 후:

- 7 inhibitors identified
- 4 compounds with sub-250 nM $IC_{50}$
- iParp1: **8.8 nM $IC_{50}$**
- protein NMR binding evidence
- **2.05 Å X-ray structure**

가 보고됩니다.

이 결과는 pure retrospective enrichment보다 훨씬 강합니다.

하지만 더 세밀하게 보면 최종 cellular utility는 완벽하지 않습니다. 논문은 iParp1의 cellular activity가 hydrolysis와 membrane permeability 문제로 biochemical potency보다 약했다고 설명합니다.

이 negative detail이 중요합니다.

$$
\text{excellent biochemical hit}
\not\Rightarrow
\text{excellent cellular drug candidate}.
$$

따라서 AdaptiveFlow는 **hit-discovery search system**의 evidence이지 complete lead-optimization solution의 evidence는 아닙니다.

---

## 14. 이 논문의 actual novelty

### 14.1 docking method 자체

새로운 universal docking architecture가 contribution의 중심은 아닙니다.

### 14.2 active learning 자체

active learning이나 surrogate filtering이라는 아이디어도 새롭지 않습니다.

### 14.3 chemical property grid 자체

physicochemical binning도 단독으로는 새로운 개념이 아닙니다.

### 진짜 contribution

이 논문의 강점은 다음 요소를 **69B-scale reproducible system**으로 조합한 것입니다.

$$
\boxed{
\text{library preparation}
+
\text{searchable chemical-space index}
+
\text{adaptive compute allocation}
+
\text{heterogeneous docking backend}
+
\text{massive orchestration}
+
\text{prospective wet validation}
}
$$

특히 ML 연구 관점에서 가장 재사용 가능한 메시지는 다음입니다.

> **expensive oracle를 더 잘 근사하는 모델만 만들지 말고, oracle call을 어디에 쓸지 학습/설계하라.**

---

## 15. 이 논문이 잘 보여주는 것

### 15.1 search policy가 scale bottleneck을 바꿀 수 있다

69B 전체를 동일 fidelity로 평가하지 않아도 target-relevant regions에 compute를 집중할 수 있습니다.

### 15.2 chemical-space organization은 model-independent leverage다

같은 grid/policy 위에서 docking engine을 바꿀 수 있으므로 adaptive selection과 oracle quality를 분리해 연구할 수 있습니다.

### 15.3 infrastructure abstraction이 ML experimentation을 쉽게 한다

AFLP / AFVS / AFU separation은 data preparation, oracle, orchestration을 modular하게 만듭니다.

### 15.4 prospective validation이 있다

FSP1/PARP1의 synthesis, biochemical assay, structural validation은 pure retrospective benchmark보다 강한 evidence입니다.

---

## 16. 이 논문이 아직 증명하지 않는 것

### 16.1 ATG-VS가 모든 target class에서 최적이다

Target pocket, scoring function, property-grid structure에 따라 optimal search policy는 달라질 수 있습니다.

### 16.2 property-nearby molecules가 activity-nearby라는 보장

Physicochemical similarity와 binding-mechanism similarity는 동일하지 않습니다.

### 16.3 optional ML이 biology를 추가로 학습했다

Classifier target이 docking-derived label이므로 scoring-function bias를 그대로 학습할 수 있습니다.

### 16.4 1,500 protocols가 동일 quality를 가진다

Protocol availability는 benchmark parity가 아닙니다.

### 16.5 nanomolar biochemical potency가 drug quality를 보장한다

PARP1 cellular result가 보여주듯 permeability, stability, metabolism 등은 별도 문제입니다.

---

## 17. 가장 중요한 ablation: fixed compute budget policy comparison

이 논문에서 가장 직접적으로 가져와서 해볼 실험은 다음입니다.

같은:

- target
- ligand library
- receptor preparation
- docking engine
- total docking-call budget
- final high-fidelity evaluation

을 고정하고 selection policy만 바꿉니다.

| Arm | Search policy |
| --- | --- |
| A | uniform random sampling |
| B | physicochemical grid + uniform tranche sampling |
| C | ATG representative prescreen → tranche selection |
| D | ATG + learned docking-score surrogate |
| E | ATG + uncertainty-aware acquisition |

Primary metrics는 하나가 아니라 vector여야 합니다.

$$
(\text{EF},\text{recall},\text{scaffold diversity},\text{oracle calls},\text{wall time},\text{wet hit rate}).
$$

가능하면 **rare scaffold recall**을 별도 metric으로 둬야 합니다.

---

## 18. 더 강한 active-search model은 어떻게 만들 수 있을까

AdaptiveFlow의 current ML filter는 docking score를 binary label로 바꾸는 relatively simple surrogate입니다.

다음 확장은 자연스럽습니다.

### 18.1 calibrated regression

$$
\hat s(m)=f_\theta(m)
$$

로 docking score 자체를 predict하고 uncertainty를 같이 추정합니다.

### 18.2 acquisition function

$$
a(m)
=
\mu_\theta(m)-\beta\sigma_\theta(m)
$$

처럼 exploitation과 exploration을 분리할 수 있습니다.

### 18.3 tranche-level uncertainty

molecule 하나가 아니라 tranche 단위 posterior를 유지합니다.

$$
q(\mathcal T_j)
=
(\mu_j,\sigma_j,n_j,\text{diversity}_j).
$$

그러면 `대표 molecule 하나가 나빴다`는 이유만으로 large diverse tranche 전체를 버리는 위험을 줄일 수 있습니다.

### 18.4 multi-fidelity oracle

cheap score, learned score, docking, rescoring, short MD, experiment를 하나의 fidelity ladder로 볼 수 있습니다.

$$
\pi:
\text{candidate state}
\rightarrow
\text{next oracle + budget allocation}.
$$

이 방향은 단순 virtual screening을 넘어 agentic scientific search와도 직접 연결됩니다.

---

## 19. 데이터와 split 관점에서 조심할 부분

AdaptiveFlow 같은 search-policy 논문은 일반 predictive model과 split leakage 형태가 다릅니다.

### Chemical-space leakage

surrogate model을 만들 때 train/eval molecules가 가까운 analog series로 겹치면 performance가 과대평가될 수 있습니다.

### Target leakage

여러 target에 걸친 generic active policy를 주장하려면 target-held-out evaluation이 필요합니다.

### Oracle leakage

surrogate가 특정 docking engine의 scores를 학습하고 동일 engine으로 평가되면 `physical generalization`이 아니라 **oracle imitation** 성능일 수 있습니다.

### Selection bias

최종 synthesized set은 이미 여러 docking / property / toxicity filters를 통과한 selected population입니다. Wet hit rate를 전체 library population의 probability처럼 해석하면 안 됩니다.

---

## 20. Reproducibility contract

AdaptiveFlow는 software release 측면에서는 강한 편입니다.

공식 organization 아래에:

- [AdaptiveFlow-LP](https://github.com/QuantumAI4Bio/AdaptiveFlow-LP)
- [AdaptiveFlow-VS](https://github.com/QuantumAI4Bio/AdaptiveFlow-VS)
- [AdaptiveFlow-Unity](https://github.com/QuantumAI4Bio/AdaptiveFlow-Unity)

가 공개되어 있고 source code는 GPL v2.0으로 제공됩니다.

AFU repository는 실제 config-driven docking workflow와 다수 docking/scoring backends를 설명합니다.

하지만 **paper-scale reproduction**은 단순 `git clone`과 다릅니다.

다음을 구분해야 합니다.

1. code path reproduction
2. small benchmark reproduction
3. 5M-scale ATG benchmark reproduction
4. 69B full screen reproduction
5. prospective synthesis/assay reproduction

특히 ready-to-dock REAL Space 자체의 data scale과 access 조건은 software reproducibility와 별개입니다.

---

## 21. Failure modes

| Failure mode | 왜 중요한가 |
| --- | --- |
| Representative miss | rare but useful chemotype가 tranche에서 관측되지 않음 |
| Property-grid bias | physicochemical bins가 activity-relevant topology를 충분히 표현하지 못함 |
| Docking-score bias | prescreen과 ML filter가 같은 oracle bias를 공유 |
| Activity cliffs | nearby descriptor region이 binding response에서 불연속 |
| Oversized tranches | 평균이 아닌 heavy-tail cell에서 representative coverage가 부족 |
| Early pruning | 한번 버린 region을 이후 다시 탐색하지 못함 |
| Protocol heterogeneity | 1,500 available protocols의 score scales / accuracy가 동일하지 않음 |
| Receptor-prep sensitivity | target preparation choice가 tranche ranking을 바꿀 수 있음 |
| I/O bottleneck | massive ready-to-dock library에서 compute보다 storage/data movement가 병목 |
| Prospective selection bias | synthesized candidates가 이미 다단계 filter를 통과한 selected set |

---

## 22. 이 논문에서 내 연구에 가져올 가장 중요한 사고방식

Structure-based drug discovery AI에서 `모델을 더 크게` 만드는 것보다 다음 질문이 더 실용적인 경우가 많습니다.

> **같은 총 계산 예산으로 어떤 molecule / target / pose에 다음 evaluation을 쓸 것인가?**

이 관점은 docking뿐 아니라 다음에도 그대로 적용됩니다.

### Pose refinement

모든 pose를 동일 steps로 refine하지 않고 uncertainty / clash / score disagreement가 큰 pose에 compute를 집중할 수 있습니다.

### Protein–ligand structure prediction

모든 complexes를 expensive recycle / diffusion sampling budget으로 처리하지 않고 difficulty-adaptive compute를 줄 수 있습니다.

### Scoring

cheap model → expensive ensemble → physics-based rescoring을 staged oracle로 구성할 수 있습니다.

### Molecular generation

generator가 수백만 samples를 만들고 모두 dock하는 대신 surrogate / diversity / uncertainty를 사용해 evaluation budget을 allocate할 수 있습니다.

즉 AdaptiveFlow의 reusable abstraction은:

$$
\boxed{
\text{generator/search space}
+
\text{cheap probe}
+
\text{adaptive selector}
+
\text{expensive oracle}
+
\text{decision rule}
}
$$

입니다.

---

## 23. 내가 추가로 요구하고 싶은 benchmark

### 23.1 fixed-dollar benchmark

`GPU-hours`보다 실제 cloud / cluster cost를 고정합니다.

### 23.2 fixed-wall-clock benchmark

동일 turnaround deadline에서 recovered hits를 비교합니다.

### 23.3 rare-chemotype challenge

active molecules를 deliberately sparse tranches / activity-cliff regions에 배치해 early pruning robustness를 봅니다.

### 23.4 cross-oracle validation

QuickVina 기반 selector가 GNINA / alternative docking / experiment에서도 enrichment를 유지하는지 봅니다.

### 23.5 target-held-out active policy

특정 target에서 tuned된 acquisition policy가 새로운 protein family에도 적용되는지 확인합니다.

### 23.6 prospective compute accounting

최종 wet hit 하나당:

$$
\text{dock calls},
\text{GPU/CPU hours},
\text{storage read},
\text{wall time},
\text{synthesized compounds}
$$

를 모두 기록하면 실제 method comparison이 훨씬 강해집니다.

---

## 24. Figure를 결과와 연결해서 읽는 법

앞의 공식 figures는 다음 claim hierarchy와 대응시켜 읽는 것이 좋습니다.

| Figure | 보여주는 것 | 보여주지 않는 것 |
| --- | --- | --- |
| Fig. 1 | platform modularity와 supported workflow surface | 각 protocol의 benchmark superiority |
| Fig. 2 | library preparation과 18D indexing idea | property grid가 activity manifold의 최적 representation이라는 증거 |
| Fig. 3 | ATG-VS search policy와 compute allocation logic | 모든 target에서 exhaustive보다 항상 우월하다는 보장 |

즉 architecture diagram을 evidence table처럼 읽지 않는 것이 중요합니다.

Prospective efficacy의 핵심 evidence는 별도의 FSP1/PARP1 assay와 structural results에 있습니다.

---

## 25. Final verdict

**Verdict: Must Read for virtual screening systems and compute-budgeted SBDD.**

AdaptiveFlow를 기억할 때 `69 billion molecules`라는 숫자만 기억하면 핵심을 놓칩니다.

더 중요한 메시지는 다음입니다.

1. **chemical space를 searchable regions로 먼저 구조화한다.**
2. **cheap target-specific probes로 promising regions를 찾는다.**
3. **expensive docking budget을 선택적으로 할당한다.**
4. **oracle 자체와 selection policy를 분리한다.**
5. **최종 평가는 enrichment에서 끝내지 않고 synthesis / assay / structure까지 올라간다.**

이 설계는 앞으로 generative molecular design이 커질수록 더 중요해질 가능성이 높습니다. 생성 모델이 10만 개가 아니라 10억 개 candidate를 쉽게 제안하게 되면 병목은 generation이 아니라 **evaluation allocation**으로 이동하기 때문입니다.

---

## 6개월 뒤 기억해야 할 세 가지

1. **AdaptiveFlow의 진짜 novelty는 docking network가 아니라 adaptive search system이다.** 18D tranche space에서 representative probes를 먼저 평가하고, target-specific evidence가 있는 region에만 비싼 계산을 집중한다.
2. **ATG-VS의 핵심 risk는 early-pruning false negatives다.** Property-nearby와 activity-nearby가 항상 같지 않으므로 rare chemotype / activity-cliff / cross-oracle recall을 별도로 봐야 한다.
3. **좋은 screening benchmark는 accuracy-only가 아니다.** Fixed compute에서 enrichment, diversity, wall time, oracle calls, prospective wet hit rate를 함께 비교해야 한다.

---

## Sources

- [Cecchini et al., Nature Biotechnology (2026), DOI 10.1038/s41587-026-03217-x](https://www.nature.com/articles/s41587-026-03217-x)
- [AdaptiveFlow GitHub organization](https://github.com/QuantumAI4Bio)
- [AdaptiveFlow-LP](https://github.com/QuantumAI4Bio/AdaptiveFlow-LP)
- [AdaptiveFlow-VS](https://github.com/QuantumAI4Bio/AdaptiveFlow-VS)
- [AdaptiveFlow-Unity](https://github.com/QuantumAI4Bio/AdaptiveFlow-Unity)
- [AdaptiveFlow preprint, bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.25.537981v3)
- [AdaptiveFlow project](https://adaptive-flow.ai/)

### Visual provenance

- Figure 1, **“The AdaptiveFlow platform for ULVSs”** — official Nature Biotechnology article, linked above; author-produced system overview.
- Figure 2, **“Organization and preparation of the Enamine REAL Space”** — official Nature Biotechnology article; author-produced library/indexing overview.
- Figure 3, **“Conceptual workflow of ATG-VSs”** — official Nature Biotechnology article; author-produced screening-policy schematic.

The figures are referenced through the official article rather than copied from third-party mirrors. Their captions in this note explain the decision-relevant interpretation; the visual itself remains attributed to the paper authors/publisher.