---
title: "Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting"
date: 2026-02-19 13:00:00 +0900
description: "CW-Gen은 조건부 평균과 공분산을 활용한 conditional whitening을 통해 시계열 diffusion/flow matching 모델의 성능을 향상시키는 프레임워크입니다."
categories: [AI, Generative Models]
tags: [diffusion, flow-matching, time-series, conditional-whitening]
math: true
mermaid: true
image:
  path: /assets/img/posts/cw-gen-framework.png
  alt: "CW-Gen framework overview"
---

## Hook

시계열 확률적 예측(probabilistic time series forecasting)은 금융, 헬스케어, 기후 예측 등 다양한 분야에서 불확실성을 정량화하는 핵심 기술이다. 최근 diffusion model과 flow matching 기반 생성 모델들이 이 분야에서 주목받고 있지만, 대부분은 **informative prior를 무시한다**. 이 논문은 간단한 질문에서 출발한다: "diffusion model의 terminal distribution을 표준정규분포 N(0,I)에서 조건부 평균과 공분산으로 parameterize된 N(μ,Σ)로 바꾸면 성능이 개선될까? 그렇다면 언제, 왜?"

## Problem

시계열 확률적 예측의 핵심은 과거 관측치 **C** ∈ ℝ^(d×T_h)가 주어졌을 때 미래 시계열 **X**_0 ∈ ℝ^(d×T_f)의 조건부 분포 P(**X**|**C**)를 학습하는 것이다. 하지만 실제 데이터는 다음과 같은 특성 때문에 학습이 어렵다:

1. **Non-stationarity**: 장기 트렌드, 계절성, heteroscedasticity(이분산성)
2. **Inter-variable dependency**: 변수 간 복잡한 상관관계
3. **Distribution shift**: 훈련 데이터와 테스트 데이터 분포 불일치

기존 diffusion model(TimeGrad, CSDI, SSSD, Diffusion-TS)들은 standard Gaussian noise N(0,I)를 terminal distribution으로 사용하며, 조건부 분포 P(**X**|**C**)를 직접 학습한다. 일부 방법들은 조건부 평균 𝔼[**X**_0|**C**]를 prior로 사용하지만(CARD, TimeDiff, TMDM), 여전히 한계가 있다:

- **TimeDiff**: linear regressor만 사용해 복잡한 패턴 포착 실패
- **TMDM**: 조건부 평균만 고려, heteroscedasticity에 취약
- **NsDiff**: 평균과 분산 regressor를 따로 학습하지만, 역과정이 복잡하고 변수 간 correlation 무시

핵심 질문은 이렇게 정리된다: **어떤 조건에서 terminal distribution을 N(0,I)에서 N(μ̂,Σ̂)로 교체하면 생성 품질이 개선되는가? 그리고 μ̂와 Σ̂를 어떻게 정확하게 추정할 것인가?**

## Key Idea

CW-Gen의 핵심 아이디어는 **conditional whitening**이다. 데이터 **X**_0에서 조건부 평균 μ̂를 빼고, 조건부 공분산의 역제곱근 Σ̂^(-0.5)를 곱하는 선형 변환이다:

$$
\textbf{X}^{\text{CW}}_0 := \widehat{\Sigma}^{-0.5}_{\textbf{X}_0|\textbf{C}} \circ (\textbf{X}_0 - \widehat{\mu}_{\textbf{X}|\textbf{C}})
$$

이 변환은 세 가지 효과를 낸다:

1. **평균 제거**: μ̂를 빼서 non-stationary trend와 계절성 제거
2. **분산 정규화**: Σ̂^(-0.5)로 heteroscedasticity 완화
3. **상관관계 제거**: 변수 간 선형 correlation 완화

Whitening된 데이터 **X**^CW_0는 **가능한 한 정상성(stationary)에 가까워지며**, diffusion model이 temporal dependency와 고차 상관관계를 더 효과적으로 학습할 수 있게 된다. 중요한 점은 이 변환이 **full-rank linear transformation**이므로 **완전히 가역적(invertible)**이라는 것이다.

이론적으로, 저자들은 **Theorem 1**에서 다음을 증명한다: terminal distribution을 N(μ̂,Σ̂)로 교체하면 KL divergence D_KL(P(**X**|**C**) ∥ N(μ̂,Σ̂))가 D_KL(P(**X**|**C**) ∥ N(0,I))보다 작아질 수 있으며, 이는 생성 품질 향상으로 이어진다. 

충분조건: μ와 Σ를 충분히 정확하게 추정하고, Σ̂의 최소 고유값을 0에서 멀리 유지하면 성능이 개선된다. 이는 total variation distance의 상한을 타이트하게 만든다.

## How It Works

![CW-Gen Framework](https://arxiv.org/html/2509.20928/x1.png)
_Figure 1: CW-Gen 전체 파이프라인. JMCE가 조건부 평균과 공분산을 추정하고, whitening → diffusion/flow → inverse whitening 과정을 거쳐 샘플 생성. 출처: 원 논문 Figure 1_

### 1. Overall Architecture

CW-Gen은 두 가지 instantiation을 제공한다:
- **CW-Diff**: Conditional Whitened Diffusion Model (SDE 기반)
- **CW-Flow**: Conditional Whitened Flow Matching (ODE 기반)

전체 파이프라인은 다음과 같다:

```mermaid
graph TD
    A[Historical Obs C] --> B[JMCE]
    B --> C[μ̂_X|C]
    B --> D[Σ̂_X0|C]
    E[Data X0] --> F[Conditional Whitening]
    C --> F
    D --> F
    F --> G[X0^CW = Σ̂^-0.5 ∘ X0 - μ̂]
    G --> H[Diffusion/Flow in Whitened Space]
    H --> I[Reverse Process]
    I --> J[X̂_τmin^CW]
    J --> K[Inverse Whitening]
    C --> K
    D --> K
    K --> L[Generated Sample X̂]
    
```

```python
# Overall Architecture Pseudocode
class CWGen:
    def __init__(self, model_type='diff'):
        self.jmce = JMCE()  # Joint Mean-Covariance Estimator
        if model_type == 'diff':
            self.generator = CWDiff()
        else:
            self.generator = CWFlow()
    
    def forward(self, C):
        # Step 1: Estimate conditional mean & covariance
        mu_hat, Sigma_hat = self.jmce(C)  # [d×Tf], [d×d×Tf]
        
        # Step 2: Conditional whitening
        X0_CW = self.conditional_whiten(X0, mu_hat, Sigma_hat)
        
        # Step 3: Diffusion/Flow in whitened space
        X_tau_CW = self.generator.forward(X0_CW, C)
        
        return X_tau_CW
    
    def sample(self, C):
        mean, cov = self.jmce(C)
        X_tau_min_CW = self.generator.reverse(C)
        X_hat = self.inverse_whiten(X_tau_min_CW, mean, cov)
        return X_hat
```

### 2. Joint Mean-Covariance Estimator (JMCE)

JMCE는 조건부 평균과 sliding-window 공분산을 **동시에(jointly)** 학습하는 novel estimator다. 왜 sliding-window인가? 진짜 조건부 공분산은 매우 복잡하고 non-smooth해서 일관된 추정이 어렵다. Sliding-window covariance는 더 정확한 근사를 제공하면서 계산 효율도 높인다.

**입력/출력:**
- 입력: 과거 관측치 **C** ∈ ℝ^(d×T_h)
- 출력: 
  - μ̂_**X**|**C** ∈ ℝ^(d×T_f) (조건부 평균)
  - L̂_1|**C**, ..., L̂_Tf|**C** (각 시점의 Cholesky factor, lower-triangular matrix)
  - Σ̂_**X**0,t|**C** := L̂_t|**C** L̂^⊤_t|**C** ∈ ℝ^(d×d) (positive semi-definite 보장)

**아키텍처:**
- Backbone: Non-stationary Transformer (Liu et al., 2022)
- Cholesky decomposition을 사용해 Σ̂가 항상 PSD임을 보장

```python
class JMCE(nn.Module):
    """Joint Mean-Covariance Estimator
    
    동시에 조건부 평균과 sliding-window 공분산을 추정하며,
    Cholesky decomposition으로 PSD를 보장.
    """
    def __init__(self, d, Tf, d_model=512, n_heads=8, n_layers=4):
        super().__init__()
        self.d = d
        self.Tf = Tf
        
        # Non-stationary Transformer backbone
        self.transformer = NonStationaryTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # 평균 예측 head
        self.mean_head = nn.Linear(d_model, d * Tf)
        
        # 공분산 예측 head (각 시점마다 lower-triangular matrix)
        # d×d lower-triangular = d(d+1)/2 parameters per time step
        self.cov_head = nn.Linear(d_model, Tf * d * (d + 1) // 2)
    
    def forward(self, C):
        # C: (batch, d, Th)
        batch_size = C.shape[0]
        
        # Transformer encoding
        h = self.transformer(C)  # (batch, d_model)
        
        # 조건부 평균 추정
        mu_hat = self.mean_head(h).view(batch_size, self.d, self.Tf)
        
        # Cholesky factors 추정
        L_params = self.cov_head(h)  # (batch, Tf * d(d+1)/2)
        L_hat = self.params_to_cholesky(L_params)  # (batch, Tf, d, d)
        
        # Σ̂ = L L^T
        Sigma_hat = torch.matmul(L_hat, L_hat.transpose(-2, -1))
        
        return mu_hat, Sigma_hat
    
    def params_to_cholesky(self, params):
        """파라미터를 lower-triangular matrices로 변환"""
        batch_size = params.shape[0]
        L = torch.zeros(batch_size, self.Tf, self.d, self.d)
        
        idx = 0
        for t in range(self.Tf):
            for i in range(self.d):
                for j in range(i + 1):
                    L[:, t, i, j] = params[:, idx]
                    idx += 1
        
        # 대각 원소는 양수로 (numerical stability)
        for i in range(self.d):
            L[:, :, i, i] = F.softplus(L[:, :, i, i]) + 1e-6
        
        return L
```

### 3. JMCE Loss Function

Theorem 1의 충분조건을 기반으로 설계된 loss는 네 가지 항의 조합이다:

$$
\mathcal{L}_{\text{JMCE}} = \mathcal{L}_2 + \mathcal{L}_{\text{SVD}} + \lambda_{\min}\sqrt{d \cdot T_f} \mathcal{L}_F + w_{\text{Eigen}} \sum_{t=1}^{T_f} \mathcal{R}_{\lambda_{\min}}(\widehat{\Sigma}_{\textbf{X}_0,t|\textbf{C}})
$$

각 항의 의미:
- **ℒ_2**: 조건부 평균 추정 오차 (MSE)
- **ℒ_F**: 공분산 Frobenius norm 오차
- **ℒ_SVD**: 공분산 nuclear norm 오차 (singular value 합)
- **ℛ_λmin**: 최소 고유값 penalty (0에서 멀리 유지)

```python
def jmce_loss(X0, C, mu_hat, Sigma_hat, lambda_min=0.1, w_eigen=50):
    """JMCE training loss
    
    Args:
        X0: (batch, d, Tf) - 실제 미래 시계열
        C: (batch, d, Th) - 과거 관측치
        mu_hat: (batch, d, Tf) - 추정 조건부 평균
        Sigma_hat: (batch, Tf, d, d) - 추정 sliding-window 공분산
        lambda_min: 최소 고유값 threshold
        w_eigen: eigenvalue penalty weight
    """
    batch_size, d, Tf = X0.shape
    
    # 1. ℒ_2: Mean estimation error
    L2 = torch.mean((X0 - mu_hat) ** 2)
    
    # 2. Sliding-window covariance 계산 (ground truth)
    Sigma_tilde = compute_sliding_window_cov(X0, window_size=95)
    
    # 3. ℒ_F: Frobenius norm error
    diff = Sigma_hat - Sigma_tilde
    LF = torch.mean(torch.norm(diff, p='fro', dim=(-2, -1)))
    
    # 4. ℒ_SVD: Nuclear norm error (sum of singular values)
    LSVD = 0
    for t in range(Tf):
        _, S, _ = torch.svd(diff[:, t])  # S: singular values
        LSVD += torch.mean(S.sum(dim=1))
    LSVD /= Tf
    
    # 5. ℛ_λmin: Eigenvalue penalty
    R_eigen = 0
    for t in range(Tf):
        eigvals = torch.linalg.eigvalsh(Sigma_hat[:, t])  # (batch, d)
        penalty = F.relu(lambda_min - eigvals)  # ReLU(λ_min - λ_i)
        R_eigen += penalty.sum(dim=1).mean()
    
    # Total loss
    loss = L2 + LSVD + lambda_min * math.sqrt(d * Tf) * LF + w_eigen * R_eigen
    
    return loss
```

### 4. Conditional Whitening Operation

JMCE가 μ̂와 Σ̂를 출력하면, 원본 데이터 **X**_0를 whitening한다:

```python
def conditional_whiten(X0, mu_hat, Sigma_hat):
    """Conditional whitening transformation
    
    X0^CW = Σ̂^(-0.5) ∘ (X0 - μ̂)
    
    Args:
        X0: (batch, d, Tf)
        mu_hat: (batch, d, Tf)
        Sigma_hat: (batch, Tf, d, d)
    """
    B, d, Tf = X0.shape
    X_centered = X0 - mu_hat  # (batch, d, Tf)
    
    # Compute Σ̂^(-0.5) via eigen-decomposition
    Sigma_inv_sqrt = []
    for t in range(Tf):
        vals, vecs = torch.linalg.eigh(Sigma_hat[:, t])
        vals_inv_sqrt = 1.0 / torch.sqrt(vals + 1e-6)
        
        # Σ^(-0.5) = Q Λ^(-0.5) Q^T
        Sigma_t_inv = vecs @ torch.diag_embed(vals_inv_sqrt) @ vecs.transpose(-2, -1)
        Sigma_inv_sqrt.append(Sigma_t_inv)
    
    Sigma_inv_sqrt = torch.stack(Sigma_inv_sqrt, dim=1)
    
    # Apply Σ̂^(-0.5) to each time step
    X0_CW = torch.zeros_like(X0)
    for t in range(Tf):
        X0_CW[:, :, t] = torch.matmul(
            Sigma_inv_sqrt[:, t], 
            X_centered[:, :, t].unsqueeze(-1)
        ).squeeze(-1)
    
    return X0_CW
```

### 5. CW-Diff: Conditionally Whitened Diffusion Model

CW-Diff는 whitened space에서 표준 DDPM을 수행한다. Forward SDE: d**X**^CW_τ = -½β_τ **X**^CW_τ dτ + √β_τ d**W**_τ, τ ∈ [0,1]. Terminal distribution은 **X**^CW_1 ≈ N(0, I) (standard Gaussian in whitened space).

Score matching loss: 𝔼[‖s_θ(α_τ **X**^CW_0 + σ_τ ε, C, τ) + ε/σ_τ‖²]로 score network를 학습한다.

```python
def cw_diff_sampling(C, score_net, mu_hat, Sigma_hat, num_steps=50):
    """CW-Diff reverse sampling
    
    Args:
        C: (batch, d, Th) - 과거 관측치
        score_net: 학습된 score network s_θ
        mu_hat, Sigma_hat: JMCE 출력
        num_steps: reverse step 수
    """
    B, d, Tf = mu_hat.shape
    
    # Step 1: Sample from terminal distribution (whitened space)
    X_CW = torch.randn(B, d, Tf)  # N(0, I)
    
    # Step 2: Reverse SDE
    dt = 1.0 / num_steps
    for step in range(num_steps, 0, -1):
        tau = step / num_steps
        alpha_tau, sigma_tau, beta_tau = get_diffusion_params(tau)
        
        # Score function
        score = score_net(X_CW, C, tau)
        
        # Reverse step
        drift = -0.5 * beta_tau * X_CW - beta_tau * score
        diffusion = math.sqrt(beta_tau) * torch.randn_like(X_CW)
        
        X_CW = X_CW + drift * dt + diffusion * math.sqrt(dt)
    
    # Step 3: Inverse whitening
    X_hat = inverse_whiten(X_CW, mu_hat, Sigma_hat)
    
    return X_hat

def inverse_whiten(X_CW, mu_hat, Sigma_hat):
    """X̂ = Σ̂^(0.5) ∘ X^CW + μ̂"""
    B, d, Tf = X_CW.shape
    Sigma_sqrt = []
    for t in range(Tf):
        vals, vecs = torch.linalg.eigh(Sigma_hat[:, t])
        Sigma_sqrt.append(
            vecs @ torch.diag_embed(torch.sqrt(vals + 1e-6)) @ vecs.T
        )
    
    X_hat = torch.zeros_like(X_CW)
    for t in range(Tf):
        X_hat[:, :, t] = Sigma_sqrt[t] @ X_CW[:, :, t].unsqueeze(-1).squeeze(-1)
    return X_hat + mu_hat
```

**핵심 차이점:**
- 기존 DDPM: **X**_0 → noise in original space
- CW-Diff: **X**^CW_0 → noise in **whitened space** (더 stationary하고 correlation이 적음)

### 6. CW-Flow: Conditionally Whitened Flow Matching

CW-Flow는 computational efficiency를 위해 ODE 기반 flow matching을 사용한다. CW-Diff는 Σ̂^(-0.5)를 계산하기 위해 eigen-decomposition이 필요해 O(d³T_f) 복잡도가 발생하는데, CW-Flow는 이를 회피한다.

Flow ODE: d**X**^CW_τ = (ε^CW - **X**_0) dτ, τ ∈ [0,1]. 여기서 ε^CW ~ N(μ̂, Σ̂) (원본 공간에서 직접 샘플링).

Vector field matching loss로 v_ψ를 학습하고, reverse ODE를 풀어 샘플을 생성한다.

```python
def cw_flow_sampling(C, vector_field, mu_hat, Sigma_hat, num_steps=50):
    """CW-Flow ODE sampling (더 효율적: Σ̂^(-0.5) 계산 불필요)"""
    B, d, T = mu_hat.shape
    epsilon_CW = sample_from_gaussian(mu_hat, Sigma_hat)
    
    X = epsilon_CW
    dt = 1.0 / num_steps
    
    # Solve ODE backward
    for step in range(num_steps, 0, -1):
        tau = step / num_steps
        v = vector_field(X, C, tau)
        X = X - v * dt  # Euler method
    
    return X  # No inverse whitening needed!

def sample_from_gaussian(mu, Sigma):
    """N(μ̂, Σ̂)에서 샘플링 (Cholesky로 효율적)"""
    B, d, Tf = mu.shape
    epsilon = torch.randn(B, d, Tf)
    
    # Σ̂^(0.5) ∘ ε + μ̂
    samples = torch.zeros_like(mu)
    for t in range(Tf):
        L = torch.linalg.cholesky(Sigma[:, t])  # Cholesky decomposition
        samples[:, :, t] = torch.matmul(
            L, epsilon[:, :, t].unsqueeze(-1)
        ).squeeze(-1) + mu[:, :, t]
    
    return samples
```

**CW-Flow의 장점:**
1. **No inverse whitening**: 샘플이 이미 원본 공간에 있음
2. **Faster**: eigen-decomposition 불필요, Cholesky만 사용 (O(d³T_f) → O(d³T_f/3))
3. **Simpler**: 1-step ODE solve vs. multi-step SDE

### 7. Training Procedure

전체 훈련은 2-stage로 진행된다:

```python
# Stage 1: Train JMCE
jmce = JMCE(d, Tf)
jmce_optimizer = torch.optim.Adam(jmce.parameters(), lr=1e-3)

for epoch in range(num_jmce_epochs):
    for X0, C in dataloader:
        mu_hat, Sigma_hat = jmce(C)
        loss = jmce_loss(X0, C, mu_hat, Sigma_hat)
        
        jmce_optimizer.zero_grad()
        loss.backward()
        jmce_optimizer.step()

# Stage 2: Train CW-Diff/CW-Flow
jmce.eval()  # freeze JMCE
cw_gen = CWDiff()  # or CWFlow()
gen_optimizer = torch.optim.Adam(cw_gen.parameters(), lr=1e-4)

for epoch in range(num_gen_epochs):
    for X0, C in dataloader:
        with torch.no_grad():
            mu_hat, Sigma_hat = jmce(C)
        
        # Conditional whitening
        X0_CW = conditional_whiten(X0, mu_hat, Sigma_hat)
        
        # Diffusion/Flow loss
        loss = cw_gen.loss(X0_CW, C)  # score matching or FM loss
        
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()
```

### 8. Why Conditional Whitening Works

Conditional whitening이 효과적인 이유는 세 가지다:

1. **Stationarity**: μ̂를 빼면 trend/seasonality 제거 → 데이터가 더 stationary해짐
2. **Homoscedasticity**: Σ̂^(-0.5)로 시간에 따른 분산 변화(heteroscedasticity) 완화
3. **Decorrelation**: 변수 간 선형 상관관계 제거 → diffusion model이 temporal dependency에 집중

이론적으로, Theorem 1은 μ̂와 Σ̂가 충분히 정확하고 Σ̂의 최소 고유값이 0에서 멀리 떨어져 있으면, terminal distribution을 N(μ̂, Σ̂)로 바꾸는 것이 KL divergence를 줄인다는 것을 보장한다. 이는 total variation distance의 상한(upper bound)을 타이트하게 만들어 생성 품질을 향상시킨다.

## Results

저자들은 5개 실세계 데이터셋(ETTh1, ETTh2, ILI, Weather, Solar Energy)에서 6개 baseline 모델(TimeDiff, SSSD, Diffusion-TS, TMDM, NsDiff, FlowTS)에 CW-Gen을 적용해 평가했다.

**평가 지표:**
- **CRPS** (Continuous Ranked Probability Score): 확률적 예측 정확도
- **QICE** (Quantile Interval Coverage Error): 예측 구간의 calibration
- **ProbCorr**: 변수 간 correlation 포착 능력
- **Conditional FID**: 생성 샘플의 전체적 품질

**주요 결과:**

| Dataset | Win Rate (CW-Gen) |
|---------|-------------------|
| ETTh1   | 83.33%           |
| ETTh2   | 88.33%           |
| ILI     | 80.00%           |
| Weather | 87.50%           |
| Solar   | 85.42%           |

Win rate는 CW-Gen이 원본 모델을 능가한 경우의 비율을 의미한다. 전체적으로 **80% 이상의 케이스에서 성능 향상**을 보였다.

구체적으로 ETTh1 데이터셋 결과를 보면:

| Model | CRPS ↓ | QICE ↓ | ProbCorr ↓ | Cond. FID ↓ |
|-------|--------|--------|------------|-------------|
| Diffusion-TS (Raw) | 0.245 | 0.089 | 0.142 | 2.87 |
| **CW-Diffusion-TS** | **0.228** | **0.081** | **0.128** | **2.54** |
| NsDiff (Raw) | 0.232 | 0.085 | 0.135 | 2.71 |
| **CW-NsDiff** | **0.219** | **0.078** | **0.121** | **2.42** |
| FlowTS (Raw) | 0.251 | 0.092 | 0.148 | 2.95 |
| **CW-FlowTS** | **0.233** | **0.084** | **0.131** | **2.61** |

모든 지표에서 CW 버전이 원본을 능가한다. 특히 **ProbCorr와 Conditional FID의 감소**는 CW-Gen이 변수 간 상관관계와 전체 분포를 더 잘 포착함을 의미한다.

![Comparison Results](https://arxiv.org/html/2509.20928/x2.png)
_Figure 2: ETTh1 데이터셋에서 Diffusion-TS, NsDiff, FlowTS와 CW 버전 비교. Prior 없는 모델은 평균/분산이 shifted되는 반면, CW 버전은 더 정확한 평균과 peak 포착을 보여준다. 출처: 원 논문 Figure 2_

**시각화 결과:** Figure 2에서 Diffusion-TS, NsDiff, FlowTS와 그들의 CW 버전을 비교한 결과, prior 없는 모델(Diffusion-TS, FlowTS)은 평균과 분산이 shifted되어 distribution shift에 취약한 반면, CW-Diffusion-TS와 CW-FlowTS는 평균 shift 없이 peak를 더 정확하게 포착했다. CW-NsDiff는 NsDiff보다 더 정확한 평균과 작은 표준편차를 보여 더 신뢰할 수 있는 불확실성 정량화를 제공한다.

## Discussion

**Strengths:**

1. **Theoretical foundation**: Theorem 1이 언제 conditional whitening이 작동하는지 명확한 충분조건 제시
2. **Joint estimation**: JMCE가 평균과 공분산을 동시에 학습하며, eigenvalue regularization으로 수치 안정성 보장
3. **Generality**: CW-Gen은 다양한 diffusion/flow matching 모델에 즉시 적용 가능
4. **Efficiency**: CW-Flow는 CW-Diff보다 계산 효율적

**Limitations (논문에서 언급):**

1. **Unfavorable regimes**: Theorem 1의 충분조건이 항상 만족되는 건 아니다. 특히 signal magnitude ‖μ‖²가 작거나, μ̂/Σ̂의 추정 오차가 크거나, 최소 고유값이 0에 가까우면 성능이 저하될 수 있다.
2. **Sliding-window approximation**: 진짜 조건부 공분산이 아니라 sliding-window covariance를 사용 → long-range correlation을 완전히 포착하지 못할 수 있음
3. **Two-stage training**: JMCE를 먼저 훈련하고 freeze → end-to-end joint training보다 덜 optimal할 수 있음

**Future directions (논문에서 제시):**

- JMCE와 generative model의 joint training
- Adaptive window size for sliding-window covariance
- Extension to irregular time series and multimodal forecasting

## Limitations

1. **충분조건이 항상 만족되지 않음**: Theorem 1의 조건(정확한 μ̂/Σ̂ 추정, 최소 고유값 > 0)이 만족되지 않으면 오히려 성능이 저하될 수 있다. Signal magnitude ‖μ‖²가 작은 데이터셋에서는 conditional whitening의 이점이 줄어든다.
2. **Sliding-window covariance 근사**: 진짜 조건부 공분산 대신 sliding-window covariance를 사용하므로, long-range temporal correlation을 완전히 포착하지 못한다.
3. **Two-stage training의 비최적성**: JMCE를 먼저 학습하고 freeze한 후 생성 모델을 학습하므로, end-to-end joint optimization 대비 suboptimal할 수 있다.
4. **계산 비용 증가**: CW-Diff는 Σ̂^(-0.5) 계산을 위해 eigen-decomposition이 필요하여 O(d³T_f) 추가 비용이 발생한다. CW-Flow가 이를 완화하지만 여전히 vanilla 모델보다 느리다.
5. **변수 차원 제한**: 변수 수 d가 클 때 d×d 공분산 행렬 추정이 불안정해질 수 있으며, 고차원 시계열에 대한 확장성이 검증되지 않았다.

## Conclusion

CW-Gen은 시계열 확률적 예측에서 diffusion/flow matching 모델의 terminal distribution을 조건부 통계량으로 parameterize하는 체계적인 방법을 제시했다. Theorem 1을 통해 conditional whitening이 언제 효과적인지에 대한 이론적 근거를 마련했고, JMCE라는 novel estimator로 조건부 평균과 공분산을 안정적으로 추정했다. 5개 데이터셋, 6개 baseline에서 80% 이상의 win rate는 이 접근법의 일반성을 보여준다. 특히 non-stationarity와 heteroscedasticity가 심한 시계열에서 가장 큰 개선을 보이며, informative prior를 활용한 생성 모델링의 가능성을 열었다.

## TL;DR

**CW-Gen**은 diffusion/flow matching 모델에 **conditional whitening**을 도입해 시계열 확률적 예측 성능을 향상시키는 프레임워크다. 핵심은 JMCE로 조건부 평균과 공분산을 정확하게 추정하고, 데이터를 whitened space로 변환해 non-stationarity, heteroscedasticity, inter-variable correlation을 완화하는 것이다. 이론적으로 KL divergence 감소를 보장하는 충분조건을 제시하며, 5개 데이터셋에서 6개 모델에 적용해 80% 이상의 케이스에서 성능 향상을 달성했다. CW-Diff(SDE 기반)와 CW-Flow(ODE 기반) 두 가지 instantiation을 제공하며, 후자가 더 효율적이다.

## Paper Info

| 항목 | 내용 |
|---|---|
| **Title** | Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting |
| **Authors** | Yanfeng Yang et al. (The Institute of Statistical Mathematics & East China Normal University) |
| **Venue** | arXiv preprint |
| **Submitted** | 2025-09 |
| **Published** | arXiv preprint, September 2025 |
| **Link** | [arXiv:2509.20928](https://arxiv.org/abs/2509.20928) |
| **Paper** | [arXiv:2509.20928](https://arxiv.org/abs/2509.20928) |
| **Code** | [GitHub](https://github.com/Yanfeng-Yang-0316/Conditionally_whitened_generative_models) |

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다. 
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
