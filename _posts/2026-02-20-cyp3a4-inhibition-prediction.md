---
title: "CYP3A4 Inhibition Prediction: Multi-Modal GNN with Molecular Fingerprints"
date: 2026-02-20 10:02:00 +0900
categories: [Projects, Drug Discovery]
tags: [gnn, cyp3a4, molecular-fingerprint, drug-metabolism, multi-modal]
math: true
---

> Dacon [Boost up AI 2025: 신약 개발 경진대회](https://dacon.io/competitions/official/236518/overview/description) 우승 모델
{: .prompt-info }

## Overview

**CYP3A4**는 전체 의약품의 약 50%를 대사하는 핵심 효소다. 신약 후보물질이 CYP3A4를 억제하면 약물-약물 상호작용(DDI)을 유발할 수 있어, 초기 단계에서 CYP3A4 억제 정도(pIC50)를 예측하는 것이 중요하다.

이 모델은 분자의 **그래프 구조**와 **9종의 molecular fingerprint**를 동시에 활용하는 multi-modal 아키텍처로, Dacon 신약 개발 경진대회에서 우승했다.

## Model Architecture

### Dual-Stream Design

```
Stream 1: SMILES → Molecular Graph → GatedGCN-LSPE (6L) → Graph Features
Stream 2: SMILES → 9 Fingerprints → MLP Encoders → Attention Fusion → FP Features
                                                                         ↓
                                      Cross-Modal Attention → Fusion Gate → pIC50
```

분자 하나에 대해 두 가지 관점의 표현을 독립적으로 학습한 뒤, cross-modal attention으로 통합한다.

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

하나의 분자에서 **9종의 fingerprint**를 추출한다:

| Fingerprint | Dimension | 특성 |
|-------------|-----------|------|
| Descriptor | 27D | 물리화학적 성질 |
| MACCS | 167D | 구조 키 |
| Morgan | 2048D | Circular fingerprint |
| Morgan Count | 2048D | 빈도 기반 |
| Feature Morgan | 2048D | Feature 기반 variant |
| RDKit | 2048D | Topological |
| Atom Pair | 2048D | 원자쌍 기술자 |
| Topological Torsion | 2048D | Torsion 기반 |
| Pharmacophore2D | 1024D | 2D 약리작용단 |

각 fingerprint는 개별 MLP encoder를 통과한 뒤, **MultiLayerAttention**으로 동적 가중합을 수행한다. 이후 mean/max/min/std 통계를 fusion network로 통합한다.

### Cross-Modal Fusion

두 스트림의 출력을 **ContextAttention** (8-head, AdaLN)으로 cross-attend하고, learnable fusion gate로 결합한다:

$$h_{final} = \alpha \cdot h_{fused} + (1 - \alpha) \cdot h_{graph}$$

여기서 $\alpha$는 학습 가능한 gate weight이다.

### Regression Head

3개의 병렬 output head로 예측한 뒤 learnable ensemble weight(softmax)로 결합한다. Output bias를 pIC50의 typical value(~8.0)로 초기화하고, tanh 기반 soft clamping으로 [3, 13] 범위를 유지한다.

## Training

```
Hidden Dimension: 768
GNN Layers: 6
Attention Layers: 6
Optimizer: AdamW (lr=1e-5, weight_decay=5e-5)
Scheduler: CosineAnnealingWarmUpRestarts (T_0=20, T_up=5)
Batch Size: 32
Validation: 5-Fold CV
Gradient Clipping: max_norm=1.0
```

### Loss Function

대회 평가 지표에 맞춘 커스텀 loss:

$$\mathcal{L} = 0.5 \times \left(1 - \min\left(\frac{\text{RMSE}}{\max(y) - \min(y)},\ 1\right)\right) + 0.5 \times r_{Pearson}$$

Normalized RMSE와 Pearson correlation을 동시에 최적화한다.

## Key Design Choices

- **Multi-modal fusion**: 그래프(3D 구조)와 fingerprint(화학적 기술자)의 상호 보완적 정보 활용
- **Domain-specific features**: CYP3A4 inhibitor/substrate SMARTS 패턴을 node feature로 직접 주입
- **9종 fingerprint ensemble**: 다양한 분자 표현을 attention으로 동적 가중하여 정보 손실 최소화
- **Multi-head output**: 3개 병렬 head의 learnable ensemble로 예측 안정성 확보
