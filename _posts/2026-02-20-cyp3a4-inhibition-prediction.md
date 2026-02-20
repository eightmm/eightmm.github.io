---
title: "CYP3A4 Inhibition Prediction: Multi-Modal GNN with Molecular Fingerprints"
date: 2026-02-20 10:02:00 +0900
categories: [Projects, ADMET]
tags: [gnn, cyp3a4, molecular-fingerprint, drug-metabolism, multi-modal, dacon]
math: true
---

> Dacon [Boost up AI 2025: 신약 개발 경진대회](https://dacon.io/competitions/official/236518/overview/description) 우승 모델
{: .prompt-info }

## Competition

### 대회 개요

**Boost up AI 2025: 신약 개발 경진대회**는 [Dacon](https://dacon.io)에서 개최된 AI 기반 신약 개발 경진대회다. 주어진 분자의 SMILES 표현으로부터 CYP3A4 효소에 대한 억제 정도(Inhibition, pIC50)를 예측하는 것이 목표다.

| 항목 | 내용 |
|------|------|
| **플랫폼** | [Dacon](https://dacon.io/competitions/official/236518/overview/description) |
| **Task** | Regression — CYP3A4 Inhibition (pIC50) 예측 |
| **Input** | Canonical SMILES |
| **Target** | Inhibition 값 (range: 0.0 ~ 99.38) |

### 데이터

| 구분 | 샘플 수 | 설명 |
|------|---------|------|
| **Train** | 1,682 | SMILES + Inhibition 값 |
| **Test** | 101 | SMILES만 제공, Inhibition 예측 |

```
ID,Canonical_Smiles,Inhibition
TRAIN_0000,Cl.OC1(Cc2cccc(Br)c2)CCNCC1,12.5
TRAIN_0001,Brc1ccc2OCCc3ccnc1c23,4.45
TRAIN_0003,Fc1ccc2nc(Nc3cccc(COc4cccc(c4)C(=O)N5CCOCC5)c3)[nH]c2c1,71.5
```

### 평가 지표

대회 평가는 **Normalized RMSE**와 **Pearson Correlation**을 동일 비중으로 결합한 커스텀 점수를 사용한다:

$$\text{Score} = 0.5 \times \left(1 - \min(A,\ 1)\right) + 0.5 \times B$$

$$A = \frac{\text{RMSE}}{\max(y) - \min(y)}, \quad B = r_{\text{Pearson}}$$

| 항목 | 설명 | 범위 |
|------|------|------|
| $A$ | RMSE를 target range(99.38)로 정규화 | [0, 1]로 clamp |
| $B$ | 예측값과 실제값의 Pearson 상관계수 | [-1, 1] |
| **Score** | 두 항의 가중합 | [0, 1], 높을수록 좋음 |

이 지표의 특성상, 단순히 오차를 줄이는 것뿐 아니라 **예측값과 실제값의 상관 경향**도 함께 최적화해야 높은 점수를 얻을 수 있다.

## Background: CYP3A4

**CYP3A4**(Cytochrome P450 3A4)는 인체 간에서 전체 의약품의 약 50%를 대사하는 핵심 효소다. 신약 후보물질이 CYP3A4를 강하게 억제하면:

- **약물-약물 상호작용(DDI)**: 병용 약물의 혈중 농도가 비정상적으로 상승
- **독성 위험 증가**: 대사되지 않은 약물이 체내 축적
- **임상 실패**: DDI 문제로 개발 중단

따라서 신약 개발 초기 단계에서 CYP3A4 억제 정도를 빠르게 스크리닝하는 것이 중요하며, 이를 AI로 예측하는 것이 이 대회의 핵심 과제다.

## Model Architecture

### Dual-Stream Design

분자 하나에 대해 **그래프 구조**와 **molecular fingerprint**라는 두 가지 관점의 표현을 독립적으로 학습한 뒤, cross-modal attention으로 통합한다.

```
Stream 1: SMILES → Molecular Graph → GatedGCN-LSPE (6L) → Graph Features
Stream 2: SMILES → 9 Fingerprints → MLP Encoders → Attention Fusion → FP Features
                                                                         ↓
                                      Cross-Modal Attention → Fusion Gate → pIC50
```

### Stream 1: Graph Feature Extractor

SMILES로부터 분자 그래프를 구성하고, GatedGCN-LSPE 6개 레이어로 처리한다.

**Node features (158D)**:
- 기본 원자 속성: atomic number, period, group, 전기음성도, degree, valence
- 화학적 속성: 혼성 궤도, formal charge, 방향족성, 키랄성
- **CYP3A4-specific features**: 13종의 inhibitor SMARTS 패턴 + 16종의 substrate SMARTS 패턴

CYP3A4에 특화된 SMARTS 패턴은 도메인 지식을 직접 feature로 주입한다:
- Inhibitor 패턴: azole 항진균제(imidazole, triazole), macrolide, HIV protease inhibitor 구조 등
- Substrate 패턴: N-dealkylation, O-dealkylation, hydroxylation site 등

**Edge features (44D)**: 결합 유형, 입체화학, 공액/고리 여부, topological distance

각 레이어에는 **Conditional Transition Block**이 추가된다:
- AdaLN (Adaptive Layer Normalization): context-dependent normalization
- SwiGLU activation으로 gating

### Stream 2: Molecular Fingerprint Extractor

하나의 분자에서 **9종의 fingerprint**를 추출하여 다양한 화학적 관점을 포착한다:

| Fingerprint | Dimension | 특성 |
|-------------|-----------|------|
| Descriptor | 27D | 물리화학적 성질 (MW, LogP, TPSA 등) |
| MACCS | 167D | 구조 키 (사전 정의된 substructure 패턴) |
| Morgan | 2048D | Circular fingerprint (원형 환경 인코딩) |
| Morgan Count | 2048D | 빈도 기반 Morgan variant |
| Feature Morgan | 2048D | Feature 기반 Morgan variant |
| RDKit | 2048D | Topological fingerprint |
| Atom Pair | 2048D | 원자쌍 기술자 |
| Topological Torsion | 2048D | Torsion 기반 구조 정보 |
| Pharmacophore2D | 1024D | 2D 약리작용단 패턴 |

각 fingerprint는 개별 MLP encoder를 통과한 뒤, **MultiLayerAttention**으로 동적 가중합을 수행한다. 9종의 fingerprint가 각각 다른 분자 특성을 포착하므로, attention을 통해 예측에 유용한 표현에 더 높은 가중치를 부여한다. 이후 mean/max/min/std 통계를 fusion network로 통합한다.

### Cross-Modal Fusion

두 스트림의 출력을 **ContextAttention** (8-head, AdaLN)으로 cross-attend하고, learnable fusion gate로 결합한다:

$$h_{final} = \alpha \cdot h_{fused} + (1 - \alpha) \cdot h_{graph}$$

여기서 $\alpha$는 학습 가능한 gate weight이다. 이를 통해 모델이 graph feature와 fingerprint feature의 기여도를 데이터로부터 학습한다.

### Regression Head

3개의 병렬 output head로 예측한 뒤 learnable ensemble weight(softmax)로 결합한다. Output bias를 pIC50의 typical value(~8.0)로 초기화하고, tanh 기반 soft clamping으로 [3, 13] 범위를 유지한다.

## Training

| 항목 | 값 |
|------|-----|
| Hidden Dimension | 768 |
| GNN Layers | 6 |
| Attention Layers | 6 |
| Optimizer | AdamW (lr=1e-5, weight_decay=5e-5) |
| Scheduler | CosineAnnealingWarmUpRestarts (T_0=20, T_up=5) |
| Batch Size | 32 |
| Epochs | 500 |
| Validation | 5-Fold CV |
| Early Stopping | patience=50 (competition score 기준) |
| Gradient Clipping | max_norm=1.0 |
| Dropout | 0.2 |

### Loss Function

대회 평가 지표를 직접 loss로 사용하여, 학습 목표와 평가 목표를 일치시킨다:

$$\mathcal{L} = -\left[0.5 \times \left(1 - \min\left(\frac{\text{RMSE}}{\max(y) - \min(y)},\ 1\right)\right) + 0.5 \times r_{Pearson}\right]$$

추가로 예측값이 유효 범위([0, 99.38])를 벗어나는 경우에 대한 range penalty를 적용한다.

### Validation Results

5-Fold Cross-Validation 결과:

| Metric | Mean ± Std |
|--------|------------|
| RMSE | 15.23 ± 1.23 |
| MAE | 11.88 ± 0.99 |
| R² | 0.765 ± 0.032 |
| Pearson Correlation | ~0.88 |

최종 제출은 5개 fold 모델의 앙상블 평균으로 생성한다.

## Key Design Choices

- **Multi-modal fusion**: 그래프(분자 구조)와 fingerprint(화학적 기술자)의 상호 보완적 정보를 cross-attention으로 통합
- **Domain-specific features**: CYP3A4 inhibitor/substrate SMARTS 패턴(29종)을 node feature로 직접 주입하여 도메인 지식 반영
- **9종 fingerprint ensemble**: 물리화학적 성질부터 약리작용단 패턴까지 다양한 분자 표현을 attention으로 동적 가중
- **Multi-head output**: 3개 병렬 head의 learnable ensemble로 예측 다양성 확보
- **Competition-aligned loss**: 평가 지표(Normalized RMSE + Pearson)를 직접 loss로 사용하여 학습-평가 목표 일치
- **5-Fold ensemble**: Cross-validation 기반 앙상블로 일반화 성능 향상
