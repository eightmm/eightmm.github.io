---
title: "Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting"
date: 2026-02-19 12:00:00 +0900
categories: [AI, Paper Review]
tags: [diffusion, flow-matching, time-series, conditional-whitening, covariance-estimation, prior]
math: true
---

## 📌 요약

Conditional whitening으로 diffusion/flow matching의 terminal distribution을 informative prior로 대체하여, 비정상 시계열 예측 성능을 일관되게 향상시키는 통합 프레임워크 **CW-Gen**을 제안한다.

> **약어:** CW-Gen, CW-Diff, CW-Flow, JMCE

## 핵심 기여

1. **통합 프레임워크 CW-Gen**: CW-Diff (Diffusion)와 CW-Flow (Flow Matching) 두 가지 instantiation 제공. CARD, TMDM, NsDiff 등 기존 방법을 특수 케이스로 포함
2. **이론적 보장**: Terminal distribution을 $N(\hat{\mu}, \hat{\Sigma})$로 대체할 때 KL divergence가 감소하는 충분조건을 수학적으로 증명 (Theorem 1, 2)
3. **JMCE (Joint Mean-Covariance Estimator)**: 조건부 평균과 sliding-window covariance를 동시 추정하는 novel estimator. Eigenvalue 제어로 안정성 확보
4. **광범위한 실증**: 5개 데이터셋 × 6개 SOTA 생성 모델에서 win rate ~76-80%

## 방법론

### Conditional Whitening

기존 diffusion의 terminal distribution $N(0, I)$를 데이터 기반 $N(\hat{\mu}_{X|C}, \hat{\Sigma}_{X|C})$로 대체:

$$X_0^{CW} := \hat{\Sigma}_{X_0|C}^{-0.5} \circ (X_0 - \hat{\mu}_{X|C})$$

- $\hat{\mu}$ 제거: 비정상 trend 및 seasonal effect 제거
- $\hat{\Sigma}^{-0.5}$ 곱하기: Heteroscedasticity 해소 및 변수 간 선형 상관관계 완화

### Theorem 1 (충분조건)

평균과 공분산을 정확하게 추정하고, 최소 eigenvalue가 충분히 크며, signal magnitude $\|\mu\|^2$가 충분히 크면 informative prior가 유리하다:

$$D_{KL}(P_{X|C} \| \hat{Q}) \leq D_{KL}(P_{X|C} \| Q_0)$$

### JMCE Loss

Theorem 1의 부등식 좌변을 minimize하도록 설계된 4개 항:
- $\mathcal{L}_2$: Mean estimation error
- $\mathcal{L}_{SVD}$: Nuclear norm for covariance
- $\mathcal{L}_F$: Frobenius norm for covariance
- $\mathcal{R}_{\lambda_{min}}$: 최소 eigenvalue penalty (수치 안정성)

### CW-Flow (효율적 버전)

CW-Diff의 $O(d^3 T_f)$ eigen-decomposition을 회피하기 위해 terminal distribution을 직접 $N(\hat{\mu}, \hat{\Sigma})$로 설정하여 ODE로 연결. Inverse matrix 계산 불필요.

## 실험 결과

| 데이터셋 | Win Rate (CRPS/QICE/ProbCorr/CondFID) | Win Rate (ProbMSE) | Win Rate (ProbMAE) |
|----------|---------------------------------------|--------------------|--------------------|
| ETTh1 | 76.0% | 75.0% | 80.0% |
| ETTh2 | 79.2% | 78.3% | 81.7% |
| ILI | 80.0% | — | — |
| Weather | 76.0% | — | — |
| Solar | 77.1% | — | — |

- 5개 데이터셋 × 6개 모델 = 30개 조합에서 **평균 ~76-80% win rate**
- ProbCorr 일관 감소: 변수 간 상관관계 캡처 능력 대폭 개선
- Distribution shift 효과적 완화

## 연구 연결점

- **물리 기반 prior 설계**: PL diffusion/flow matching에서도 binding pocket geometry, pharmacophore 등을 informative prior로 활용 가능
- **Covariance modeling**: Atom 간 거리/각도 분포의 conditional covariance 학습으로 realistic 3D conformation 생성
- **Terminal distribution 설계의 새로운 패러다임** 제시

## 링크

- 📄 [arXiv: 2509.20928](https://arxiv.org/abs/2509.20928)
- 💻 [GitHub](https://github.com/Yanfeng-Yang-0316/Conditionally_whitened_generative_models)
