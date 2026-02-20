---
title: "Out-of-Distribution Detection in Molecular Complexes via Diffusion Models for Irregular Graphs"
date: 2026-02-19 14:00:00 +0900
categories: [AI, Drug Discovery]
tags: [protein-ligand, diffusion, binding-affinity]
math: true
---

## 📌 요약

PF-ODE 기반 diffusion 모델로 3D 분자 그래프의 OOD 탐지를 수행한 **최초의 연구**. Trajectory 기반 18개 geometric feature로 complexity bias를 극복하고, GEMS binding affinity 모델 오류와 강한 상관관계를 보인다.

> **중요도:** ⭐⭐⭐⭐ | **약어:** PF-ODE OOD

## 핵심 기여

1. 3D 기하학적 그래프를 위한 **최초의 unsupervised OOD detection framework**
2. 3D 좌표(연속)와 원자/잔기 타입(이산)을 단일 연속 공간에서 동시 처리하는 통합 diffusion 모델
3. PF-ODE trajectory 기반 **18개 geometric feature** — complexity bias 극복
4. GEMS binding affinity 모델 오류와의 강한 상관관계: $R^2$ r=0.750, MAE r=-0.880
5. Proposition 2.1: Likelihood가 high probability로 prediction error를 제어함을 증명

## 방법론

### 3단계 파이프라인

1. **Unified Continuous Diffusion** 학습 — Categorical features를 spherical embedding 후 3D 좌표와 concat, SE(3)-equivariant GNN (EGNN) 사용
2. **PF-ODE**로 exact log-likelihood 계산:

$$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla \cdot v_t(x_t) \, dt$$

3. **18개 trajectory feature** 추출 + Gaussian KDE LDR classifier

### 18개 Trajectory Features

| 카테고리 | 주요 Feature |
|----------|-------------|
| Geometric Inefficiency | Path tortuosity, efficiency |
| Local Instability | Max Lipschitz estimate |
| Vector Field Activity | VF mean/max/std, spikiness, acceleration |
| Energetic Cost | Total flow energy |
| Feature-Coordinate Coupling | Coupling consistency |

> **핵심 인사이트**: ID sample은 효율적이고 직선적인 trajectory, OOD sample은 erratic하고 chaotic한 경로를 보인다.

## 실험 결과

### Complexity Bias 발견 및 극복

- 3dd0 (α-carbonic anhydrase)은 OOD임에도 training set보다 **높은 likelihood** → low structural complexity가 원인
- Trajectory features 추가 시 3dd0을 성공적으로 OOD로 분류

### GEMS Error 예측

| 상관관계 | Pearson r |
|----------|-----------|
| Median log-likelihood ↔ GEMS $R^2$ | 0.750 |
| Median log-likelihood ↔ GEMS MAE | -0.880 |

- Low likelihood → large GEMS error (exponential 관계)
- **실용적 의의**: New sample의 likelihood로 예측 신뢰도 사전 판단 가능

### 데이터셋

PDBbind v2020에서 7개 protein family를 완전히 제외한 strict OOD split 사용 (총 19,443개 complex).

## 한계

- Intermediate OOD에서 낮은 정확도
- PF-ODE trajectory 계산의 computational overhead (~5s/sample)
- 코드 미공개

## 연구 연결점

- ✅ PF-ODE trajectory feature를 **Flow Matching에 적용** 가능
- ✅ Protein-ligand binding affinity 예측에 **OOD filtering 통합**
- ✅ SE(3)-equivariant 모델의 **reliability 평가** 도구

## 링크

- 📄 [arXiv: 2512.18454](https://arxiv.org/abs/2512.18454)
