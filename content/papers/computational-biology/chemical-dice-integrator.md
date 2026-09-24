---
title: Chemical Dice Integrator — Multimodal Molecular Teacher, Sequence-Distilled Student
aliases:
  - papers/chemical-dice-integrator
tags:
  - papers
  - computational-biology
  - molecular-representation
  - multimodal-learning
  - knowledge-distillation
  - mamba
  - smiles
  - property-prediction
status: full-note
source_type: Journal
source_url: https://doi.org/10.1038/s41467-026-77700-z
---

# Chemical Dice Integrator: Multimodal Molecular Teacher, Sequence-Distilled Student

> **한 줄 요약:** Chemical Dice Integrator(CDI)의 가장 재사용 가치가 높은 아이디어는 “여섯 molecular descriptor를 한꺼번에 쓰자”가 아니라, **비싼 multimodal chemistry stack은 teacher latent를 만들 때만 사용하고, 실제 배포에서는 SMILES 하나만 받는 sequence model이 그 latent를 직접 예측하도록 distill한다**는 분리다. Rich representation을 만드는 비용과 deployable representation의 비용을 같은 모델에 강제로 묶지 않는 설계다.

## 왜 이 논문을 저장하는가

Molecular representation learning에서는 거의 항상 같은 trade-off가 나온다.

```text
cheap representation
  → fast / broad coverage
  → but limited inductive bias

rich representation
  → graph + descriptors + quantum + bioactivity + image + language
  → but expensive / brittle / incomplete coverage
```

Fingerprint, graph encoder, SMILES language model, quantum descriptor, bioactivity signature는 서로 다른 chemical information을 본다. 한 가지 representation을 고르면 deployment는 단순하지만 특정 task에 필요한 정보가 빠질 수 있다. 반대로 모든 feature pipeline을 매 inference마다 계산하면 information coverage는 넓어져도 대규모 screening에는 비싸고, 일부 featurizer가 특정 molecule에서 실패하는 순간 전체 pipeline이 깨질 수 있다.

CDI는 이 문제를 두 단계로 분리한다.

$$
\boxed{
\text{rich multimodal teacher}
\quad\rightarrow\quad
\text{cheap single-input student}
}
$$

Training-time에는 여러 chemistry expert가 만든 view를 통합한다. Deployment-time에는 그 expert들을 다시 실행하지 않고 molecular string만으로 통합 latent를 예측한다.

이 pattern은 특정 paper를 넘어 재사용 가치가 크다. Expensive 3D/quantum teacher를 cheap 2D/sequence student로 압축하거나, pocket-aware teacher를 ligand-only prescreening student로 distill하거나, assay-rich teacher를 structure-only encoder로 바꾸는 식으로 확장할 수 있기 때문이다.

두 번째 저장 이유는 **representation quality와 representation availability를 동시에 평가**한다는 점이다. 좋은 embedding이더라도 source featurizer가 일부 molecule에서 실패하면 실제 screening coverage가 깨진다. CDI-Generalised는 valid molecular string만 있으면 teacher latent의 proxy를 만들기 때문에 “성능”과 “coverage”를 같은 deployment contract 안에서 생각하게 한다.

세 번째는 evidence boundary가 비교적 명확하다. Peer-reviewed Nature Communications article은 classification/regression benchmark, scaffold-based evaluation, low-data setting, featurizer failure coverage, 그리고 yeast damage-response assay까지 연결한다. 하지만 이 evidence는 **protein–ligand binding, target-disjoint generalization, therapeutic efficacy**를 직접 증명하지 않는다.

---

## Metadata and public artifacts

| Field | Value |
| --- | --- |
| Paper | Scalable molecular representations enabled by multimodal fusion and sequence distillation |
| Authors | Suvendu Kumar, Saveena Solanki, Mudit Gupta, Sonam Chauhan, Sanjay Kumar Mohanty, et al. |
| Journal | Nature Communications |
| Published | 2026-09-19 |
| DOI | [10.1038/s41467-026-77700-z](https://doi.org/10.1038/s41467-026-77700-z) |
| Earlier preprint | Chemical Dice Integrator (CDI): A Scalable Framework for Multimodal Molecular Representation Learning, bioRxiv 2025.11.11.687860 |
| Main representation | 8192-D integrated molecular embedding |
| Teacher | CDI-Basic, hierarchical multimodal autoencoder fusion |
| Student | CDI-Generalised, Mamba-based SMILES → CDI embedding model |
| Modalities | Mordred, GROVER, ImageMol, Signaturizer, MOPAC, ChemBERTa |
| Official code | [the-ahuja-lab/ChemicalDice](https://github.com/the-ahuja-lab/ChemicalDice) |
| Code snapshot inspected | [`b64ce09cab7da073d20596668b3602d41a0c1f0c`](https://github.com/the-ahuja-lab/ChemicalDice/commit/b64ce09cab7da073d20596668b3602d41a0c1f0c) |
| Public model artifact | [the-ahuja-lab/ChemicalDice](https://huggingface.co/the-ahuja-lab/ChemicalDice) |
| Code/model license | MIT |
| Article state inspected | peer-reviewed Nature Communications publication |

> **Claim boundary:** benchmark superiority, scaffold/low-data behavior, and yeast validation are author-reported results from the paper. The existence and structure of the public implementation/model artifacts were checked separately. This note does not claim an independent benchmark reproduction.

---

## Visual guide — architecture는 이 그림 하나가 핵심을 잘 보여준다

결과 leaderboard screenshot보다 **teacher → student data flow**가 이 paper의 durable mechanism을 설명하는 데 더 중요하다. Official repository는 MIT license이고 architecture overview asset을 함께 배포하므로, 이 note에서는 commit-pinned project visual 하나만 사용한다.

![Chemical Dice Integrator official project overview](https://raw.githubusercontent.com/the-ahuja-lab/ChemicalDice/b64ce09cab7da073d20596668b3602d41a0c1f0c/Images/CDI.png)

*Source: the-ahuja-lab/ChemicalDice, `Images/CDI.png`, commit `b64ce09c…`, MIT license. 이 그림은 CDI의 multimodal integration concept를 설명하는 official project visualization이다. Paper benchmark를 독립적으로 검증하는 evidence가 아니다. Journal article과 technical docs는 six-view formulation을 canonical scientific claim으로 사용하며, current README 일부 문구가 five-view라고 남아 있는 documentation drift는 아래 reproducibility section에서 별도로 다룬다.*

---

## 1. Problem: “가장 좋은 molecular representation”은 하나가 아닐 수 있다

Molecule $m$에 대해 여러 representation function을 생각해보자.

$$
x^{(q)} = f_q(m)
$$

where $q$ indexes a modality.

CDI가 사용하는 six-view formulation은 다음과 같다.

| View | Public implementation | 주로 담는 정보 |
| --- | --- | --- |
| Physicochemical | Mordred | topology, counts, geometry-derived descriptors, chemistry statistics |
| Graph/topology | GROVER | atom/bond connectivity and pretrained graph context |
| 2D image | ImageMol | rendered molecular-image representation |
| Bioactivity | Signaturizer | pretrained bioactivity-profile representation |
| Quantum | MOPAC | semi-empirical electronic/energetic descriptors |
| Molecular language | ChemBERTa | SMILES sequence semantics |

각 view는 inductive bias가 다르다. Graph model은 adjacency와 local motifs를 자연스럽게 보지만 electronic observable을 직접 계산하지 않는다. Quantum descriptor는 전자구조 정보를 주지만 계산 비용이 크다. Bioactivity representation은 biological prior를 강하게 담을 수 있지만 upstream database coverage의 영향을 받는다. Molecular language model은 scalable하지만 input string representation에 의존한다.

따라서 문제를

$$
\text{Which single representation is best?}
$$

로 두기보다

$$
\text{Can complementary views define a richer teacher space?}
$$

와

$$
\text{Can that space later be approximated cheaply?}
$$

로 분리하는 것이 CDI의 핵심이다.

---

## 2. Core idea: rich teacher와 cheap student를 분리한다

CDI는 operationally 두 모델로 볼 수 있다.

```text
                 ┌─ Mordred
                 ├─ GROVER
molecule ────────┼─ ImageMol
                 ├─ Signaturizer
                 ├─ MOPAC
                 └─ ChemBERTa
                         │
                         ▼
                    CDI-Basic
              multimodal teacher space
                         │
             teacher embedding target
                         │
SMILES ─────────────► CDI-Generalised
                    Mamba student
                         │
                         ▼
                    8192-D embedding
```

Deployment에서 중요한 것은 아래 path다.

$$
\text{SMILES}
\rightarrow
\hat z_{CDI}
$$

즉 inference-time representation cost가 source modality 수에 비례하지 않는다.

일반적인 multimodal model은 train과 inference에서 모두 같은 modality set을 요구하는 경우가 많다. CDI에서는 multimodal view가 **teacher target을 정의하는 training resource**이고, student의 deployment input은 single modality다.

---

## 3. CDI-Basic: leave-one-view-out cross-modal commonality를 먼저 학습한다

Official implementation documentation은 CDI-Basic을 two-tier hierarchical autoencoder로 설명한다.

### Tier 1 — Semantic Commonality Autoencoders

각 modality $j$에 대해 다른 view들을 입력으로 묶는다.

$$
X_{-j}
=
\operatorname{concat}\{x^{(k)}: k\neq j\}.
$$

그 다음 encoder $E_j$가 latent를 만든다.

$$
h_j = E_j(X_{-j}).
$$

이 설계의 목적은 단순 concatenation이 아니다. `MOPAC 정보만 있는 차원`, `GROVER 정보만 있는 차원`을 그대로 쌓기보다 **다른 modalities가 한 modality와 공유하는 정보를 cross-modal reconstruction/alignment를 통해 끌어내는 것**에 가깝다.

```text
late concatenation
    = keep views side by side

CDI Tier 1
    = learn cross-view commonality
```

### 구현 문서의 주의점

Current repository의 high-level architecture page와 code-oriented technical documentation 사이에는 Tier-1 reconstruction target을 설명하는 notation이 완전히 동일하지 않은 부분이 있다. High-level description은 “other five views → omitted sixth view”를 강조하고, technical documentation 일부 식은 autoencoder reconstruction term을 다른 방식으로 표기한다.

따라서 이 note에서는 **leave-one-view-out cross-modal commonality**라는 architectural contract까지만 강하게 해석하고, exact loss bookkeeping은 pinned implementation을 기준으로 재현해야 한다. 이런 문서 불일치는 reproducibility에서 사소하지 않다. Architecture name보다 실제 training code가 최종 contract다.

---

## 4. Tier 2 — latent subspaces를 하나의 Super Embedding으로 압축한다

Tier 1 outputs를 concatenation한다.

$$
H=[h_1;h_2;\dots;h_6].
$$

그리고 Super-Embedding Autoencoder가 이를 최종 latent $z$로 압축한다.

$$
z=E_{SEA}(H),
\qquad
z\in\mathbb{R}^{8192}.
$$

이 8192-D vector가 downstream representation interface가 된다.

중요한 것은 “8192”라는 숫자 자체보다 **representation boundary가 고정된다**는 것이다. Upstream modalities는 서로 dimension과 preprocessing이 다르지만 SEA 이후에는 하나의 fixed-width vector contract로 바뀐다.

이렇게 하면 downstream model은 source featurizer의 세부사항을 알 필요가 없다.

$$
\hat y=g_\phi(z).
$$

Representation provider와 task head의 interface가 분리되는 셈이다.

---

## 5. CDI-Generalised: multimodal knowledge를 SMILES student로 distill한다

CDI-Basic은 rich하지만 deployment에는 불편하다. 새 molecule 하나를 넣을 때마다 MOPAC calculation, Mordred descriptors, graph encoder, 2D image rendering + image encoder, bioactivity signature model, molecular language model을 모두 실행하면 throughput이 낮고 failure surface도 넓다.

그래서 CDI-Generalised는 teacher embedding $z$를 target으로 삼아 molecular string에서 직접 예측한다.

$$
\hat z_\theta=f_\theta(s),
$$

where $s$ is a SMILES representation.

Official public implementation docs는 student backbone을 Mamba State-Space Model로 설명한다. Distillation의 최소 objective는 embedding regression이다.

$$
\mathcal L_{MSE}
=
\frac{1}{ND}
\sum_{i=1}^{N}
\sum_{d=1}^{D}
(z_{id}-\hat z_{id})^2.
$$

Current technical documentation은 MSE 외 angular/cosine alignment도 설명한다. Conceptually는 다음처럼 이해할 수 있다.

$$
\mathcal L_{distill}
=
\lambda_{mse}\|z-\hat z\|_2^2
+
\lambda_{ang}\bigl(1-\cos(z,\hat z)\bigr).
$$

Exact weights/config는 paper prose보다 pinned code/checkpoint를 기준으로 확인해야 한다.

### 왜 Mamba인가

이 paper에서 Mamba 자체가 핵심 novelty는 아니다. Student는 molecular string을 scalable하게 처리하는 sequence encoder 역할을 한다. Transformer student로 바꿔도 “multimodal teacher → single-input deployable student”라는 핵심 실험 질문은 유지된다.

$$
\text{student backbone choice}
<
\text{definition of teacher target}
$$

이라는 관점이 더 중요하다.

---

## 6. What is actually new: multimodality보다 train/deploy asymmetry가 더 중요하다

“여러 molecular features를 합친다”는 아이디어 자체는 새롭지 않다. 이 paper의 durable novelty를 세 층으로 나누는 편이 정확하다.

### A. Cross-view teacher construction

Simple concatenation/PCA가 아니라 cross-modal autoencoding으로 shared information을 학습한다.

### B. Distilled deployment interface

Rich teacher를 그대로 production model로 쓰지 않고 sequence-only student로 latent geometry를 전달한다.

### C. Representation utility를 coverage와 같이 본다

일부 source pipeline이 실패해도 valid molecular string에 대해 student embedding을 만들 수 있다.

이 중 가장 재사용 가치가 높은 것은 **B**다. Model design 관점에서 중요한 질문은 “최종 deployment input이 모든 training-time information source를 실제로 다시 제공할 필요가 있는가?”이다. CDI의 답은 `no`다. Training-time privilege를 latent target으로 압축할 수 있다면 inference path를 훨씬 싸게 만들 수 있다.

---

## 7. Training data와 upstream priors를 representation 자체와 구분해야 한다

Source modalities 자체도 독립적으로 pretrained knowledge를 가진다. Public implementation documentation 기준으로 GROVER는 large unlabeled molecular corpus의 graph prior를, ChemBERTa는 large SMILES corpus의 language prior를, Signaturizer는 broad bioactivity prior를, ImageMol은 pretrained image prior를 가져온다.

따라서 CDI teacher가 배우는 것은 “raw molecule에서 처음부터 모든 chemistry를 학습”하는 것이 아니다.

$$
z_{CDI}
=
F(\text{multiple pretrained/expert representations of }m).
$$

이것은 강점이면서 evaluation risk다. Downstream benchmark와 upstream pretraining source 사이 overlap이 있을 수 있기 때문이다. 특히 bioactivity-derived representation은 task label과 의미적으로 가까운 prior를 가질 가능성이 있다.

중요한 구분은

$$
\text{legitimate pretrained prior}
\neq
\text{benchmark leakage}
$$

이지만, 둘을 분리하려면 provenance audit가 필요하다.

- downstream molecules가 teacher training corpus에 있었는가?
- Signaturizer upstream supervision이 benchmark assay와 겹치는가?
- test molecule의 close analog가 upstream model pretraining에 있었는가?
- representation model이 frozen이어도 test label과 가까운 side information이 들어가는가?

Paper가 scaffold split을 제공하더라도 **downstream train/test scaffold separation은 upstream pretraining overlap과 다른 문제**다.

---

## 8. Evaluation unit: representation benchmark는 downstream evaluator까지 contract다

Earlier public preprint는 23 classification datasets의 171 tasks와 10 regression datasets에 대한 broad benchmark를 보고했다. Peer-reviewed Nature Communications paper는 CDI가 classical fusion methods보다 우수하고 established molecular descriptors와 같거나 나은 결과를 보였다고 요약한다.

이 결과를 읽을 때 representation 비교의 unit을 명확히 해야 한다.

$$
\text{representation}
+
\text{pooling/readout}
+
\text{downstream predictor}
+
\text{split}
+
\text{selection rule}
$$

이 전부가 실제 evaluation protocol이다.

Representation $z$가 달라도 downstream model capacity가 충분히 크면 차이가 줄어들 수 있고, 반대로 작은 evaluator는 latent geometry의 linear separability를 더 강하게 평가한다. 그래서 “CDI embedding이 더 좋다”는 claim은 어떤 downstream model을 썼는지, tuning budget이 동일했는지, representation dimension 차이를 어떻게 처리했는지, random/scaffold split 중 무엇인지와 함께 읽어야 한다.

---

## 9. Scaffold split evidence: useful하지만 broad OOD의 끝은 아니다

Public OOD module은 Bemis–Murcko scaffold group을 split unit으로 사용한다.

$$
g(m_i)=g(m_j)
\Rightarrow
s(m_i)=s(m_j).
$$

이는 random split보다 meaningful하다. Close scaffold family를 train/test에 흩뿌리는 shortcut을 줄이기 때문이다. Peer-reviewed article도 scaffold-based evaluation에서 stable generalization을 보고한다.

하지만 supported claim은 정확히 **new scaffold groups under the evaluated downstream protocol**이다. 아래 claim은 별도 evidence가 필요하다.

- unseen protein target generalization
- new assay/source generalization
- temporal chemistry generalization
- new modality/domain generalization
- protein–ligand OOD generalization

특히 SBDD로 가져갈 경우 scaffold split만으로는 부족하다.

$$
\text{ligand scaffold OOD}
\neq
\text{target-family OOD}.
$$

[[concepts/evaluation/scaffold-split|Scaffold split]]과 [[concepts/sbdd/protein-ligand-split|Protein–ligand split]]을 함께 봐야 한다.

---

## 10. Low-data result의 의미: representation prior가 sample efficiency를 바꾼다

Peer-reviewed article은 low-data evaluations에서 improved utility를 보고한다. Frozen representation $z=f(m)$ 위에 작은 task head를 fit한다고 하자.

$$
\hat y=g_\phi(z).
$$

Label 수가 적을 때 downstream head가 새 chemistry를 처음부터 학습할 수 없다. Teacher distillation으로 형성된 latent에 useful structure가 이미 있다면 sample complexity가 줄 수 있다.

따라서 low-data study가 테스트하는 것은 대략

$$
I(z;y)
\quad\text{is usable with small }n
$$

이다.

다만 low-data advantage가 broad transferable prior인지 보려면 label fraction만 줄였는지, scaffold diversity도 같이 줄었는지, tuning budget은 같은지, upstream corpus와 test chemistry overlap은 어떤지, class imbalance가 작은 $n$에서 더 심해지지는 않는지 확인해야 한다.

---

## 11. Coverage is a first-class metric — 하지만 coverage와 fidelity는 다르다

Nature Communications paper가 강조하는 결과 중 하나는 **individual feature-generation pipeline이 실패해도 embedding coverage를 유지한다**는 것이다.

Teacher pipeline coverage를

$$
C_T=P(\text{all required source featurizers succeed})
$$

라고 하고 student coverage를

$$
C_S=P(\text{valid SMILES accepted by student})
$$

라고 하면 student는 더 넓은 operational input domain을 가질 수 있다.

하지만

$$
\text{embedding exists}
\not\Rightarrow
\text{missing expert information is faithfully recovered}.
$$

Student가 vector를 반환했다는 사실은 coverage다. 그 molecule에서 MOPAC/Signaturizer가 실제로 제공했을 정보까지 정확히 복원했다는 증거는 아니다.

따라서 failure subset에서 별도 fidelity audit가 필요하다.

1. source featurizer가 성공하는 molecule에서 teacher–student distance 측정
2. source featurizer가 실패하는 molecule에서 downstream task quality 측정
3. failure cause별 subgroup 분석
4. uncertainty/OOD score 측정

이 네 항목을 분리해야 `complete embedding coverage`가 `complete reliability`로 과대해석되지 않는다.

---

## 12. Ablation: modality 수가 줄면 무엇을 잃는가

Official repository는 six-view model에서 modality를 순차적으로 제거하는 ablation workflow를 공개한다. Current docs에서는 MOPAC, Mordred, GROVER, Signaturizer 등을 차례로 제거하고 reconstruction-loss-based diagnostic을 기록한다.

이 ablation이 지지하는 가장 안전한 interpretation은 **modalities가 완전히 redundant하지 않으며, full teacher space가 complementary priors를 활용한다**는 것이다.

다만 “MOPAC이 가장 중요하다” 같은 ranking은 조심해야 한다. Removal order가 fixed라면 marginal contribution은 conditional하다.

$$
\Delta_j=Q(M)-Q(M\setminus\{j\})
$$

는 남아 있는 modality set $M$에 의존한다.

진짜 modality attribution을 하려면 Shapley-like subset evaluation이나 최소한 여러 removal orders가 낫다.

$$
\phi_j
\approx
\mathbb E_S[Q(S\cup\{j\})-Q(S)].
$$

따라서 current ablation은 “여러 view가 useful하다”는 evidence에는 적합하지만, modality importance의 universal ordering에는 부족하다.

---

## 13. Prospective evidence: yeast damage-response assay는 무엇을 증명하는가

Peer-reviewed article은 CDI representation을 활용해 genome-stability-protective compounds를 prioritize한 뒤 **isoeugenol**과 **eugenyl acetate**를 yeast damage-response assay에서 experimental validation했다고 보고한다.

이것은 pure retrospective benchmark보다 강한 evidence다. Pipeline level에서는

```text
representation
  → downstream predictor
  → candidate prioritization
  → wet-lab assay
```

까지 연결되기 때문이다.

하지만 supported endpoint를 정확히 유지해야 한다. 이 experiment는 yeast system의 DNA-damage/genome-stability response에 대한 prospective biological signal을 지지한다. 다음은 직접 지지하지 않는다.

- human efficacy
- clinical benefit
- protein–ligand binding affinity
- selectivity
- target mechanism
- broad drug-discovery success rate

따라서 experimental validation을 “representation이 therapeutic molecules를 발견했다”로 확대하면 안 된다.

---

## 14. 가장 중요한 confounder: teacher에 들어간 bioactivity prior

CDI의 강점 중 가장 흥미로운 modality는 Signaturizer이고, 동시에 가장 조심해서 읽어야 할 modality이기도 하다. Bioactivity embedding은 molecular structure만으로부터 얻는 neutral geometric descriptor가 아니다. Upstream bioactivity databases와 learned association을 포함한다.

따라서 downstream task가 biological activity와 가까울수록 useful prior가 커질 수 있지만 provenance sensitivity도 커진다.

필요한 audit는 다음이다.

- Signaturizer pretraining targets와 downstream benchmark target overlap
- compound identity/near-neighbor overlap
- assay-family overlap
- temporal ordering
- Signaturizer를 제거했을 때 target-disjoint performance 변화

이것은 “leakage가 있다”는 주장 아니다. 현재 public evidence만으로 leakage를 단정할 수 없다. 정확한 conclusion은 **bioactivity-informed representation의 OOD claim은 upstream provenance까지 포함해 audit해야 한다**는 것이다.

---

## 15. Reproducibility: artifacts는 강하지만 documentation drift가 있다

이 paper는 public reproducibility surface가 좋은 편이다.

확인 가능한 artifact는 다음과 같다.

- public GitHub source
- MIT license
- training/evaluation documentation
- scaffold/OOD utilities
- ablation workflow
- public Hugging Face feature-extraction model
- Python/R deployment interfaces
- peer-reviewed article supplementary/source-data surface

하지만 current repository에는 작은 documentation drift가 있다. Journal article과 architecture/technical docs는 **six modalities**를 canonical하게 설명한다. 반면 current top-level README의 일부 문장은 아직 **five modalities**라고 적는다.

이런 mismatch는 model science의 큰 오류는 아니지만 재현자가 어떤 configuration이 paper model인지 판단할 때 중요하다. 따라서 reproducible run은 최소한 다음을 pin해야 한다.

```text
paper version / DOI
+ Git commit
+ checkpoint/model revision
+ modality configuration
+ preprocessing/canonicalization
+ split definition
+ downstream evaluator
```

이 note에서 implementation reference로 고정한 Git commit은 `b64ce09c…`다.

---

## 16. What is unsupported — 이 paper에서 넘어가면 안 되는 주장

### “Multimodal student가 여섯 modality의 모든 정보를 복원한다”

Embedding alignment와 downstream utility는 information-equivalence proof가 아니다.

### “Scaffold split이 broad OOD를 증명한다”

Scaffold novelty는 target/assay/time novelty와 다르다.

### “Complete coverage는 complete reliability다”

Student가 output vector를 만드는 것과 그 vector가 trustworthy한 것은 별개다.

### “Wet-lab hit가 protein–ligand representation quality를 증명한다”

Yeast genome-stability phenotype은 SBDD binding endpoint가 아니다.

### “Six modalities가 항상 single best representation보다 낫다”

Task-specific specialist가 더 유리할 수 있다. CDI의 강한 value proposition은 cross-task robustness와 deployment simplicity다.

---

## 17. 가장 decision-useful한 추가 실험

### Experiment A — teacher → student OOD retention ratio

같은 downstream evaluator를 고정하고 다음을 비교한다.

```text
A: best single modality
B: CDI-Basic teacher
C: CDI-Generalised student
D: simple concatenation / projection baseline
```

각 split에서

$$
R_{OOD}
=
\frac{Q_{student}-Q_{single}}
{Q_{teacher}-Q_{single}}
$$

를 보면 student가 teacher의 OOD advantage를 얼마나 보존하는지 직접 측정할 수 있다.

중요 split은 random, scaffold-disjoint, similarity-cluster-disjoint, temporal, assay/source-disjoint, target/family-disjoint다.

### Experiment B — privileged modality distillation

Teacher에만 특정 high-cost modality를 추가한다.

```text
teacher: SMILES + graph + quantum + 3D
student: SMILES only
```

그리고 quantum/3D teacher prior가 student OOD에 얼마나 남는지 측정한다. 이것이 CDI 아이디어를 3D molecular AI로 확장하는 가장 직접적인 실험이다.

### Experiment C — upstream provenance audit

Signaturizer 포함/제외 teacher를 같은 downstream target-disjoint split에서 비교한다.

```text
with bioactivity prior
vs
without bioactivity prior
```

Performance difference를 target-family similarity별로 stratify하면 useful transfer와 target leakage risk를 분리할 수 있다.

### Experiment D — failure-subset audit

Teacher featurizer가 실패하는 molecule만 별도 subset으로 모은다. 그 위에서 student confidence, nearest-neighbor distance, property prediction error, scaffold novelty를 측정한다. “coverage advantage”가 가장 필요한 subset에서 실제 quality가 유지되는지 확인할 수 있다.

---

## 18. Protein–ligand pretraining에 가져갈 때의 가장 중요한 변형

CDI의 pattern을 protein–ligand problem으로 옮길 때 teacher modality를 그대로 복사할 필요는 없다. 더 중요한 것은 **privileged-information teacher**라는 구조다.

예를 들어:

```text
Teacher
  ligand 2D graph
  ligand conformer ensemble
  protein sequence
  pocket geometry
  interaction fingerprints
  assay context
        ↓
    rich latent

Student
  ligand graph/SMILES + protein sequence
        ↓
 deployable latent
```

이 setup은 expensive pocket/3D/assay information을 pretraining target으로만 쓰고, deployment input이 제한된 상황에서도 그 prior를 얼마나 보존할 수 있는지 시험한다.

PL에서는 split contract를 더 강하게 해야 한다.

- ligand scaffold split
- protein family split
- target–ligand pair split
- temporal split
- assay/source split

그렇지 않으면 student가 “3D interaction knowledge를 distill했다”가 아니라 known target/chemistry neighborhood를 압축했을 가능성을 배제하기 어렵다.

---

## 19. Complexity: 왜 teacher/student separation이 실제로 scalable한가

Teacher inference cost를 각 modality cost의 합으로 단순화하면

$$
C_T
\approx
\sum_{q=1}^{Q} C_q
+
C_{fusion}.
$$

여기에는 MOPAC 같은 expensive computation과 여러 pretrained encoders가 포함된다. Student는

$$
C_S\approx C_{seq}.
$$

Library size를 $M$이라 하면 전체 featurization cost gap은

$$
M(C_T-C_S)
$$

가 된다.

따라서 screening library가 커질수록 distillation의 경제성이 커진다. 이 구조는 quantum calculations, conformer ensemble generation, docking-derived fingerprints, large ensemble encoders처럼 “training 때는 유용하지만 매 molecule마다 반복하기 비싼” feature와 특히 잘 맞는다.

---

## 20. Falsification: 어떤 결과가 나오면 CDI story를 약하게 봐야 하나

좋은 representation story는 반증 조건이 있어야 한다. 다음 결과가 나오면 strong interpretation은 약해진다.

1. target-disjoint/temporal split에서 CDI advantage가 사라진다.
2. Signaturizer를 제거하면 biological tasks의 gain 대부분이 사라지고, 그 gain이 upstream target overlap과 강하게 연동된다.
3. Student가 IID에서는 teacher를 잘 모방하지만 scaffold/temporal OOD에서 teacher latent geometry를 보존하지 못한다.
4. Simple concatenation + matched-capacity student가 CDI-Basic과 동일한 성능을 낸다.
5. Failure subset에서 student coverage는 높지만 error/uncertainty가 크게 악화된다.
6. 8192-D representation advantage가 dimension-matched projection이나 larger downstream evaluator에서 사라진다.

반대로 위 controls를 통과하면 **multimodal privileged teacher → single-input student**는 강한 molecular representation primitive가 된다.

---

## 21. Reproducibility checklist

| Item | Status |
| --- | --- |
| Peer-reviewed article | yes |
| Permanent DOI | yes |
| Official source repository | yes |
| Public model artifact | yes |
| Code/model license | MIT |
| Six source modalities identified | yes |
| Teacher architecture documented | yes |
| Student architecture documented | yes |
| Scaffold/OOD workflow public | yes |
| Ablation workflow public | yes |
| Paper-vs-current-repo documentation drift | present; pin revision |
| Independent reproduction in this note | no |
| Target-disjoint PL evidence | not established |

---

## 22. Related notes

- [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/sbdd/protein-ligand-split|Protein–ligand split]]
- [[papers/generative-models/molexar|Molexar]]

Molexar처럼 여러 molecular conditions/modalities를 한 model interface에서 다루는 접근과 비교하면 CDI의 차이가 선명하다. Molexar 쪽은 **one model consumes many conditions at inference**에 가깝고, CDI는 **many training-time views define one latent, then one cheap input reproduces it at inference**에 가깝다.

---

## Final verdict

CDI를 “여섯 descriptor를 합친 더 큰 fingerprint”로만 읽으면 가장 중요한 design pattern을 놓친다.

이 paper의 durable insight는

$$
\boxed{
\text{train with richer information than you can afford at deployment}
}
$$

이다.

Rich multimodal representation을 최종 serving architecture로 고정하지 않고, **teacher latent geometry로 바꾼 뒤 cheap student가 그 space를 직접 예측하게 한다.** 이 때문에 expensive chemistry priors를 pretraining에 활용하면서도 large-library inference는 sequence-only path로 단순화할 수 있다.

동시에 가장 큰 unresolved question도 명확하다. Student가 teacher의 **IID average performance**가 아니라 teacher의 **true OOD advantage**까지 보존하는가? 특히 bioactivity prior와 broad pretraining corpus가 있는 상황에서는 scaffold split만으로 충분하지 않다.

따라서 이 paper를 다음 연구에 가져갈 때 핵심 metric은 단순 benchmark average보다

$$
\text{teacher-to-student OOD retention}
$$

이 되어야 한다.

---

## Three durable takeaways

1. **Training-time modality와 deployment-time modality는 같을 필요가 없다.** Expensive quantum, bioactivity, graph, image priors를 teacher target에만 쓰고 cheap sequence student로 distill하는 것은 scalable molecular representation의 강한 설계 pattern이다.

2. **Coverage와 fidelity를 분리해서 평가해야 한다.** Student가 valid SMILES 전체에 embedding을 제공하는 것은 operational advantage지만, source featurizer가 실패하는 chemistry에서 teacher information이 정확히 복원됐다는 뜻은 아니다.

3. **OOD claim은 downstream split뿐 아니라 upstream provenance까지 포함해야 한다.** Scaffold split은 useful하지만 bioactivity/large-corpus pretrained teacher에서는 target, assay, temporal, upstream-overlap audit가 함께 있어야 representation transfer를 제대로 해석할 수 있다.

---

## Sources

- Kumar, S., Solanki, S., Gupta, M. et al. [Scalable molecular representations enabled by multimodal fusion and sequence distillation](https://doi.org/10.1038/s41467-026-77700-z). *Nature Communications* (2026), published 2026-09-19.
- Earlier preprint: [Chemical Dice Integrator (CDI): A Scalable Framework for Multimodal Molecular Representation Learning](https://doi.org/10.1101/2025.11.11.687860).
- Official implementation: [the-ahuja-lab/ChemicalDice](https://github.com/the-ahuja-lab/ChemicalDice), inspected at commit [`b64ce09c…`](https://github.com/the-ahuja-lab/ChemicalDice/commit/b64ce09cab7da073d20596668b3602d41a0c1f0c).
- Official public model artifact: [the-ahuja-lab/ChemicalDice on Hugging Face](https://huggingface.co/the-ahuja-lab/ChemicalDice).
