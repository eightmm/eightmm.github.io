---
title: "Improved Protein Structure Prediction Using Potentials from Deep Learning"
date: 2026-02-20 11:00:00 +0900
description: "AlphaFold 1이 CASP13에서 deep learning 기반 distogram 예측과 gradient descent로 단백질 구조 예측의 새로운 패러다임을 제시한 방법을 자세히 분석한다."
categories: [AI, Protein Structure]
tags: [protein-structure, AlphaFold, distance-prediction, ResNet, CASP13, deep-learning]
math: true
mermaid: true
image:
  path: /assets/img/posts/alphafold1-improved-protein-structure-prediction/fig2.png
  alt: "AlphaFold 1의 folding 과정 (CASP13 target T0986s2)"
---

> 이 글은 AlphaFold 시리즈의 첫 번째 글이다. 시리즈 구성: AlphaFold 1 (이 글), AlphaFold 2, AlphaFold 3, 시리즈 정리.
{: .prompt-info }

## Hook

단백질 구조 예측은 생물학의 grand challenge였다. 50년간 수많은 이론적 시도가 있었지만, 실험 구조만큼 정확한 예측은 드물었다. 2018년 CASP13에서, DeepMind의 AlphaFold는 free modelling (FM) category에서 2위와 압도적 격차(52.8 vs 36.6 summed z-score)로 우승하며 단백질 구조 예측의 판도를 바꿨다. 

Fragment assembly와 simulated annealing이 지배하던 분야에서, AlphaFold는 deep learning으로 학습한 단백질별 potential을 gradient descent로 최적화하는 완전히 새로운 접근을 제시했다. 이전에는 불가능했던 새로운 fold들을 높은 정확도로 예측하는 시대가 열렸다.

## Problem

기존의 free modelling 접근법들은 크게 두 가지 한계를 가지고 있었다.

### 1. Fragment Assembly의 비효율성

가장 성공적인 FM 방법들(Rosetta, QUARK 등)은 fragment assembly에 의존했다. 이 방법은 PDB 구조들에서 추출한 통계적 potential을 사용하여, simulated annealing 같은 stochastic sampling으로 구조를 만들어낸다. 문제는 구조 가설을 반복적으로 수정하며 낮은 potential 구조를 찾기 위해 수천 번의 move가 필요하고, 이를 여러 번 반복해야 low-potential 구조들을 충분히 탐색할 수 있다는 점이다. 

계산 비용이 크고, 전역 최적해를 찾는다는 보장도 없다.

### 2. Contact Prediction의 제한적 정보

최근 몇 년간 evolutionary covariation을 사용한 contact prediction이 구조 예측 정확도를 개선했다. MSA에서 두 residue 위치의 상관관계 변화를 분석해 contact (Cβ atoms가 8 Å 이내) 여부를 예측하고, 이를 statistical potential에 반영하여 folding 과정을 guide한다.

하지만 contact prediction은 binary 정보다. "8 Å 이내인가, 아닌가"만 알 수 있다. 4 Å과 7.9 Å은 모두 contact지만, 구조적 의미는 완전히 다르다. 더 정확한 구조를 만들려면 더 세밀한 정보가 필요하다.

## Key Idea

AlphaFold는 두 가지 핵심 아이디어로 이 문제를 해결한다.

### Idea 1: Distogram — Distance Distribution Prediction

Binary contact 대신, **모든 residue pair의 거리 분포(distance distribution)**를 예측한다. 2-22 Å 범위를 64개 bin으로 나눠, 각 bin에 대한 확률 분포를 출력하는 것이 distogram이다. 

이는 contact prediction보다 훨씬 풍부한 정보를 제공한다. 4 Å과 7 Å을 구분할 수 있고, 예측의 불확실성(분포의 넓이)도 모델링한다. 또한 많은 거리를 동시에 예측하면서, network가 covariation, local structure, nearby residue의 identity 정보를 전파하고 통합할 수 있다.

### Idea 2: Gradient Descent Structure Realization

Distogram 예측으로부터 단백질별 potential $V_{\text{total}}(\phi, \psi)$를 구성하고, 이를 **gradient descent로 직접 최적화**한다. Fragment assembly나 stochastic sampling 없이, 미분 가능한 potential을 backbone torsion angles $(\phi, \psi)$에 대해 greedy하게 최소화한다.

초기화만 여러 번 바꿔가며 gradient descent를 반복하면, 수백 번의 iteration만으로 낮은 potential의 정확한 구조에 수렴한다. 계산 효율성과 구조 품질을 동시에 달성하는 우아한 해법이다.

## How It Works

AlphaFold의 전체 파이프라인은 크게 세 단계로 나뉜다: (1) MSA 구성 및 feature 추출, (2) Distogram 예측, (3) Structure realization.

```mermaid
graph TD
    A[Amino acid sequence S] --> B["MSA construction / HHblits + PSI-BLAST"]
    B --> C["Feature extraction / Profile, Covariation, Potts"]
    C --> D["Deep ResNet / 220 residual blocks"]
    D --> E[Distogram P_d_ij|S, MSA]
    D --> F["Torsion distributions / P_φ_i,ψ_i|S, MSA"]
    E --> G[Distance potential V_distance]
    F --> H[Torsion potential V_torsion]
    G --> I["Combined potential / V_total = V_dist + V_torsion + V_vdW"]
    H --> I
    I --> J["Gradient descent / L-BFGS on φ,ψ"]
    J --> K[Realized structure x = Gφ,ψ]
    K --> L[Repeat with noisy restarts]
    L --> M[Select lowest-potential structure]
    
```

### 4.1 Overall Pipeline

전체 시스템의 흐름을 pseudocode로 나타내면 다음과 같다.

<details markdown="1">
<summary>📝 Overall AlphaFold Pipeline Pseudocode (클릭하여 펼치기)</summary>

```python
class AlphaFold:
    def __init__(self):
        self.distogram_net = DistogramNetwork()  # 220 residual blocks
        self.torsion_net = TorsionNetwork()      # Same network, different head
    
    def predict_structure(self, sequence: str) -> Structure:
        # Step 1: MSA construction
        msa = build_msa(sequence)  # HHblits + PSI-BLAST
        
        # Step 2: Feature extraction
        features = extract_features(sequence, msa)
        # - Profile: PSI-BLAST (21), HHblits (22), non-gapped (21)
        # - Covariation: Potts model parameters (484), Frobenius norm (1)
        # - Gap/deletion features
        
        # Step 3: Distogram and torsion prediction
        distogram = self.distogram_net(features)  # L×L×64 bins (2-22 Å)
        torsion_dist = self.torsion_net(features)  # L×1296 bins (φ,ψ)
        
        # Step 4: Construct protein-specific potential
        V_distance = self.build_distance_potential(distogram)
        V_torsion = self.build_torsion_potential(torsion_dist)
        V_total = V_distance + V_torsion + V_vdW  # Rosetta score2_smooth
        
        # Step 5: Structure realization by gradient descent
        structures = []
        for _ in range(num_restarts):
            # Initialize from predicted torsion distributions
            phi, psi = sample_from(torsion_dist)
            
            # Gradient descent (L-BFGS)
            phi, psi = optimize(V_total, phi, psi, method='L-BFGS')
            
            # Convert torsions to 3D coordinates
            structure = geometry_builder(phi, psi)
            structures.append((V_total(phi, psi), structure))
        
        # Step 6: Noisy restarts from low-potential pool
        pool = sorted(structures)[:20]  # Keep 20 lowest-potential
        for _ in range(num_noisy_restarts):
            potential, structure = random.choice(pool)
            phi, psi = structure.torsions + noise(30°)  # Add 30° noise
            phi, psi = optimize(V_total, phi, psi, method='L-BFGS')
            structure = geometry_builder(phi, psi)
            structures.append((V_total(phi, psi), structure))
        
        # Return lowest-potential structure
        return min(structures, key=lambda x: x[0])[1]
```

</details>

### 4.2 MSA Construction and Feature Representation

입력은 amino acid sequence $S$다. 먼저 Uniclust30 database에서 HHblits로 유사 서열을 검색하여 MSA를 구성한다 (3 iterations, E-value = $10^{-3}$). 추가로 PSI-BLAST로 nr dataset을 검색한다.

MSA로부터 다음 feature들을 추출한다:

**1차원 features (residue 당):**
- One-hot amino acid type (21)
- Profile features: PSI-BLAST profile (21), HHblits profile (22), non-gapped profile (21), HMM profile (30)
- Potts model bias (22)
- Deletion probability (1)
- Residue index (5 bits + scalar)

**2차원 features (residue pair 당):**
- Potts model parameters (484): MSA로부터 regularized pseudolikelihood로 학습한 covariation 정보
- Frobenius norm of Potts parameters (1)
- Gap matrix (1)

총 ~650개의 feature가 각 64×64 crop에 입력된다. MSA 깊이(Neff, effective number of sequences)가 클수록 covariation signal이 강해져 distogram 정확도가 올라간다.

> MSA subsampling (절반만 사용)과 coordinate noise 추가를 data augmentation으로 사용하여, shallow MSA에서도 robust하게 예측하고 overfitting을 방지한다.
{: .prompt-tip }

### 4.3 Distance Prediction Neural Network

Distogram을 예측하는 neural network는 **220 residual blocks로 구성된 deep 2D convolutional network**다. 이전 contact prediction 연구들은 1D embedding 후 2D network를 사용했지만, AlphaFold는 처음부터 끝까지 2D로 처리한다.

#### Architecture Details

각 residual block은 다음 구조를 갖는다:

<details markdown="1">
<summary>📝 Residual Block Architecture (클릭하여 펼치기)</summary>

```python
class ResidualBlock(nn.Module):
    """AlphaFold distogram prediction residual block
    
    220 blocks total:
    - 7 groups × 4 blocks with 256 channels
    - 48 groups × 4 blocks with 128 channels
    Cycling through dilations: 1, 2, 4, 8
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.projection1 = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.dilated_conv = nn.Conv2d(
            channels, channels, 
            kernel_size=3, 
            dilation=dilation,  # 1, 2, 4, or 8
            padding=dilation     # Keep spatial dimensions
        )
        
        self.bn3 = nn.BatchNorm2d(channels)
        self.projection2 = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, 64, 64)
        residual = x
        
        x = self.bn1(x)
        x = F.elu(x)
        x = self.projection1(x)
        
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dilated_conv(x)
        
        x = self.bn3(x)
        x = F.elu(x)
        x = self.projection2(x)
        
        return x + residual  # Skip connection


class DistogramNetwork(nn.Module):
    """Full distogram prediction network"""
    def __init__(self):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv2d(num_features, 256, kernel_size=1)
        
        # 7 groups × 4 blocks with 256 channels
        self.blocks_256 = nn.ModuleList([
            ResidualBlock(256, dilation=(i % 4) * 2 + 1)  # 1,2,4,8 cycle
            for i in range(7 * 4)
        ])
        
        # 48 groups × 4 blocks with 128 channels
        self.downsample = nn.Conv2d(256, 128, kernel_size=1)
        self.blocks_128 = nn.ModuleList([
            ResidualBlock(128, dilation=(i % 4) * 2 + 1)
            for i in range(48 * 4)
        ])
        
        # Output head: distance distribution (64 bins)
        self.output = nn.Conv2d(128, 64, kernel_size=1)
        # Position-specific bias: indexed by residue offset (capped at 32)
        self.position_bias = nn.Parameter(torch.randn(32, 64))
    
    def forward(self, features: Tensor) -> Tensor:
        # features: (batch, num_features, 64, 64)
        x = self.input_proj(features)
        
        # 256-channel blocks
        for block in self.blocks_256:
            x = block(x)
        
        # Downsample to 128 channels
        x = self.downsample(x)
        
        # 128-channel blocks
        for block in self.blocks_128:
            x = block(x)
        
        # Output: distance distribution
        logits = self.output(x)  # (batch, 64, 64, 64) - last 64 is bins
        
        # Add position-specific bias
        for i in range(64):
            for j in range(64):
                offset = min(abs(i - j), 31)
                logits[:, :, i, j] += self.position_bias[offset, :]
        
        # Softmax over bins
        distogram = F.softmax(logits, dim=1)  # (batch, 64_bins, 64, 64)
        return distogram
```

</details>

**핵심 설계:**
- **Dilated convolutions**: dilation을 1, 2, 4, 8로 순환하며 정보를 빠르게 전파. 64×64 crop에서 멀리 떨어진 residue pair 간에도 정보 교환 가능.
- **Deep architecture**: 220개 residual blocks가 복잡한 covariation 패턴과 local structure 제약을 학습.
- **2D throughout**: 1D embedding 없이 처음부터 2D feature map으로 처리하여 spatial correlation을 최대한 활용.

출력은 $L \times L \times 64$ 크기의 distogram으로, 각 residue pair $(i,j)$에 대해 64개 거리 bin (2-22 Å)의 확률 분포 $P(d_{ij} | S, \text{MSA}(S))$를 나타낸다.

### 4.4 Cropped Distograms and Ensembling

메모리 제약과 overfitting 방지를 위해, network는 항상 **64×64 crop**에서 학습하고 테스트한다. 하나의 단백질로부터 수천 개의 다른 crop을 생성할 수 있어 강력한 data augmentation 효과를 낸다.

전체 $L \times L$ distogram을 예측하려면:
1. 여러 offset으로 64×64 crop을 tile하여 전체 거리 행렬을 커버
2. 각 crop의 예측을 평균 (crop 중앙부에 높은 가중치)
3. 독립적으로 학습한 4개 모델의 예측을 ensemble

이렇게 구성한 distogram은 높은 정확도와 불확실성 모델링을 보인다 (Fig. 3). 예측 분포의 표준편차가 낮을수록 실제 거리와의 오차가 작다.

### 4.5 Potential Construction

Distogram과 torsion distribution으로부터 미분 가능한 potential을 구성한다.

#### Distance Potential

각 거리 분포를 cubic spline으로 보간하여 smooth function을 만들고, negative log probability를 합산한다:

$$
V_{\text{distance}}(\phi, \psi) = \sum_{i < j} -\log P(d_{ij}(\phi, \psi) | S, \text{MSA}(S))
$$

여기서 $d_{ij}(\phi, \psi) = \|x_i(\phi, \psi) - x_j(\phi, \psi)\|$는 torsion angles로부터 계산한 Cβ 좌표 간 거리다.

**Reference distribution correction**: 단순히 negative log probability를 쓰면 prior가 과대표현된다. 서열과 무관하게 단백질 길이만으로 학습한 reference distribution $P(d_{ij}|\text{length})$를 빼서 보정한다:

$$
V_{\text{distance}} = \sum_{i < j} \left[ -\log P(d_{ij} | S, \text{MSA}) + \log P(d_{ij} | \text{length}) \right]
$$

이는 log-likelihood ratio 형태로, sequence-specific information만 남긴다.

#### Torsion Potential

Network의 별도 output head는 각 residue의 $(\phi_i, \psi_i)$ marginal distribution을 1296개 bin (10° 간격)으로 예측한다. 이를 unimodal von Mises distribution으로 fitting하여:

$$
V_{\text{torsion}}(\phi, \psi) = \sum_i -\log P(\phi_i, \psi_i | S, \text{MSA}(S))
$$

#### Combined Potential

최종 potential은 세 항의 합이다:

$$
V_{\text{total}}(\phi, \psi) = V_{\text{distance}} + V_{\text{torsion}} + V_{\text{vdW}}
$$

여기서 $V_{\text{vdW}}$는 Rosetta의 score2_smooth van der Waals term으로 steric clash를 방지한다. Cross-validation 결과 세 항에 equal weighting을 적용하는 것이 가장 좋았다.

### 4.6 Structure Realization by Gradient Descent

Potential이 미분 가능하므로, backbone torsion angles $(\phi, \psi)$를 변수로 gradient descent를 적용한다.

<details markdown="1">
<summary>📝 Gradient Descent Structure Realization (클릭하여 펼치기)</summary>

```python
def realize_structure(distogram, torsion_dist, sequence):
    """Realize protein structure by gradient descent
    
    Args:
        distogram: L×L×64 distance distribution predictions
        torsion_dist: L×1296 torsion angle distribution predictions
        sequence: amino acid sequence (length L)
    
    Returns:
        Best structure (lowest potential)
    """
    L = len(sequence)
    
    # Build differentiable potentials
    V_distance = build_distance_potential(distogram, sequence)
    V_torsion = build_torsion_potential(torsion_dist)
    V_vdW = lambda phi, psi: rosetta_score2_smooth(phi, psi)
    
    def V_total(phi, psi):
        return V_distance(phi, psi) + V_torsion(phi, psi) + V_vdW(phi, psi)
    
    # Pool of low-potential structures
    pool = []
    
    # Phase 1: Initial sampling from predicted torsion distributions
    for restart in range(500):
        # Sample initial torsions from von Mises fitted distributions
        phi_init = sample_von_mises(torsion_dist[:, :18])  # φ
        psi_init = sample_von_mises(torsion_dist[:, 18:])  # ψ
        
        # Gradient descent with L-BFGS
        phi, psi = lbfgs_optimize(
            V_total, 
            x0=(phi_init, psi_init),
            max_iter=1200,
            tolerance=1e-5
        )
        
        # Build 3D structure from optimized torsions
        structure = geometry_builder(phi, psi, sequence)
        potential = V_total(phi, psi)
        
        pool.append((potential, structure))
        pool = sorted(pool)[:20]  # Keep 20 lowest
    
    # Phase 2: Noisy restarts from pool
    for restart in range(4500):
        if random.random() < 0.9:
            # 90%: noisy restart from pool
            _, structure = random.choice(pool)
            phi_init, psi_init = structure.torsions
            # Add 30° noise
            phi_init += np.random.normal(0, 30°, size=L)
            psi_init += np.random.normal(0, 30°, size=L)
        else:
            # 10%: fresh sample from torsion distributions
            phi_init = sample_von_mises(torsion_dist[:, :18])
            psi_init = sample_von_mises(torsion_dist[:, 18:])
        
        # Gradient descent
        phi, psi = lbfgs_optimize(
            V_total, 
            x0=(phi_init, psi_init),
            max_iter=1200
        )
        
        structure = geometry_builder(phi, psi, sequence)
        potential = V_total(phi, psi)
        
        pool.append((potential, structure))
        pool = sorted(pool)[:20]
    
    # Return lowest-potential structure
    best_potential, best_structure = pool[0]
    return best_structure


def lbfgs_optimize(V, x0, max_iter=1200, tolerance=1e-5):
    """L-BFGS optimization of torsion angles
    
    Each step:
    1. Compute V(φ, ψ) and gradients ∇_φ V, ∇_ψ V
    2. Update φ, ψ with L-BFGS step
    3. Check convergence
    """
    phi, psi = x0
    
    for step in range(max_iter):
        # Compute potential and gradients
        potential = V(phi, psi)
        grad_phi = gradient(V, phi, wrt='phi')
        grad_psi = gradient(V, psi, wrt='psi')
        
        # L-BFGS update (maintains history of gradients)
        phi, psi = lbfgs_step(phi, psi, grad_phi, grad_psi)
        
        # Check convergence
        if np.linalg.norm([grad_phi, grad_psi]) < tolerance:
            break
    
    return phi, psi


def geometry_builder(phi, psi, sequence):
    """Build 3D coordinates from torsion angles
    
    Uses ideal bond lengths and angles:
    - N-Cα: 1.46 Å
    - Cα-C: 1.52 Å
    - C-N: 1.33 Å
    - Bond angles: N-Cα-C = 110°, Cα-C-N = 117°
    """
    coords = []
    # Initialize first residue at origin
    coords.append(np.array([0, 0, 0]))  # N
    
    for i, aa in enumerate(sequence):
        # Build backbone atoms using φ, ψ
        N = coords[-1]
        Ca = N + rotation(phi[i]) @ np.array([1.46, 0, 0])
        C = Ca + rotation(psi[i]) @ np.array([1.52, 0, 0])
        
        # Cβ (or Cα for glycine)
        if aa == 'G':
            Cb = Ca
        else:
            Cb = Ca + np.array([0, 1.52, 0])  # Simplified
        
        coords.extend([Ca, C, Cb])
    
    return Structure(coords, sequence)
```

</details>

**Gradient Descent 과정 (Fig. 2c 참조):**
1. **Initialization**: Predicted torsion distribution에서 $(\phi, \psi)$ sampling
2. **Optimization**: L-BFGS로 $V_{\text{total}}$를 1200 steps 최적화
3. **Pooling**: 낮은 potential의 구조 20개를 pool에 유지
4. **Noisy restarts**: Pool에서 선택한 구조에 30° noise를 추가해 재최적화 (90%), 또는 fresh sampling (10%)
5. **Convergence**: 수백 번 반복 후 lowest-potential 구조 선택

각 gradient descent step은 greedy하게 potential을 낮추지만, 전역적인 conformational change를 일으켜 잘 packing된 구조로 수렴한다. Noisy restart 덕분에 fresh sampling보다 높은 TM score를 달성한다 (평균 0.641 vs 0.636).

> Gradient descent는 simulated annealing보다 훨씬 빠르다. 수백 번의 restart로 수렴하는 반면, fragment assembly는 수천-수만 번의 move가 필요하다.
{: .prompt-tip }

### 4.7 Training and Auxiliary Losses

Network는 cross-entropy loss로 학습한다:

$$
\mathcal{L}_{\text{distance}} = -\sum_{i,j} \log P(d_{ij}^{\text{true}} | S, \text{MSA}(S))
$$

여기서 $d_{ij}^{\text{true}}$는 PDB 구조의 실제 Cβ 거리가 속한 bin이다.

추가로 auxiliary losses를 사용하여 one-dimensional representation을 개선한다:
- **Secondary structure prediction**: 8-class DSSP labels (weight 0.005)
- **Accessible surface area**: Relative ASA prediction (weight 0.001)

이 auxiliary heads는 2D activation을 mean/max pooling하여 1D로 변환 후 예측한다. Secondary structure Q3 accuracy 84%로 state-of-the-art 수준이다.

**Training setup:**
- Batch size: 4 crops × 8 GPUs = 32
- Optimizer: Synchronized SGD with 0.85 dropout
- Learning rate: 0.06, decayed by 50% at 150k, 200k, 250k, 350k steps
- Training time: 5 days for 600k steps

### 4.8 Full Chains Without Domain Segmentation

긴 단백질은 전통적으로 domains로 분할하여 독립적으로 folding했다. 하지만 domain segmentation 자체가 어렵고 error-prone하다.

AlphaFold는 **전체 chain을 한 번에 folding**한다. Sliding window 방식으로 여러 크기(64, 128, 256 residues)의 subsequence MSA를 계산하고, 각각의 distogram을 평균하여 full-chain distogram을 만든다. MSA 깊이로 가중 평균하면, alignment가 많은 region에서 더 정확한 예측을 얻는다.

이 방식은 domain boundary를 모르는 상황에서도 전체 구조를 예측할 수 있게 한다.

## Results

AlphaFold는 CASP13에서 압도적 성능을 보였다.

### Free Modelling Performance

| Metric | AlphaFold | 2nd Place (Group 322) |
|--------|-----------|----------------------|
| **FM summed z-score** | **52.8** | 36.6 |
| **FM+FM/TBM z-score** | **68.3** | 48.2 |
| FM domains with TM > 0.6 | **22** | 10 |

AlphaFold는 FM category에서 2위보다 **44% 높은 점수**를 기록했다. 특히 0.6-0.7 TM score 범위에서 다른 모든 시스템을 압도하며, 이전에는 불가능했던 정확도의 새로운 fold 예측들을 생산했다 (Fig. 1a).

![AlphaFold CASP13 performance](/assets/img/posts/alphafold1-improved-protein-structure-prediction/fig1.png)
_Figure 1: (a) FM domains predicted at given TM-score threshold. AlphaFold가 0.6-0.7 범위에서 압도적. (b) 새로운 6개 fold에 대한 TM score 비교. (c) Long-range contact prediction precision — AlphaFold distogram이 최고 정확도._

### Contact Prediction Accuracy

Distogram을 8 Å threshold로 binary contact prediction으로 변환하면, long-range contact prediction에서도 state-of-the-art를 달성한다 (Fig. 1c). Top L, L/2, L/5 contacts에서 모두 highest precision을 기록했다.

이는 distogram이 풍부한 정보를 담고 있어, 단순히 thresholding해도 기존 contact prediction 전용 방법들을 능가함을 보여준다.

### Distogram Accuracy and Structure Quality

Distogram lDDT (DLDDT12)와 realized structure의 TM score 간 강한 상관관계가 있다 (Pearson r = 0.92, Fig. 4a). 즉, distogram 자체가 정확하면 최종 구조도 정확하다.

![Distogram accuracy vs TM score](/assets/img/posts/alphafold1-improved-protein-structure-prediction/fig4.png)
_Figure 4: (a) TM score vs distogram lDDT — 높은 상관관계. (b) Potential의 각 component를 제거했을 때 TM score 변화 — distance potential이 가장 중요._

Distance potential을 완전히 제거하면 TM score가 0.266으로 떨어진다 (Fig. 4b). Torsion potential, reference correction, van der Waals term은 각각 소폭 기여하지만, distance potential이 압도적으로 중요하다.

### Template-Based Modelling

AlphaFold는 FM 방법임에도 TBM category에서도 강력한 성능을 보였다. Assessors' formula로 **TBM top-one에서 4위, best-of-five에서 1위**를 차지했다. Template 없이도 homology modeling 수준의 정확도에 도달할 수 있음을 시사한다.

## Discussion

AlphaFold는 protein structure prediction에서 새로운 패러다임을 제시했지만, 논문은 몇 가지 한계와 향후 방향을 밝히고 있다.

### MSA Depth Dependency

Distogram 정확도는 MSA의 effective number of sequences (Neff)에 크게 의존한다. Shallow MSA (Neff가 낮은 경우)에서는 covariation signal이 약해 예측 정확도가 떨어진다. MSA subsampling augmentation으로 어느 정도 완화했지만, orphan proteins이나 최근에 발견된 서열은 여전히 어렵다.

### FM vs TBM Gap

FM 성능이 크게 향상되었지만, TBM에 비하면 여전히 gap이 있다. 논문은 "FM targets still lag behind TBM targets and cannot yet be relied on for detailed understanding of hard structures"라고 밝혔다. Side-chain configuration이나 binding site의 세밀한 구조까지 신뢰하기는 어렵다.

### Gradient Descent Local Minima

Gradient descent는 local minima에 빠질 수 있다. Noisy restart로 어느 정도 해결하지만, 매우 복잡한 topology를 가진 단백질에서는 global optimum을 찾지 못할 가능성이 있다. 논문은 "no guarantee of finding global optimum"을 인정한다.

### Biological Applications

논문은 AlphaFold 예측이 biological insights를 제공할 수 있는 수준에 도달하기 시작했다고 주장한다. Contact predictions만으로도 mutation targeting에 유용하고, 예측 구조가 protein-protein interface prediction, binding pocket identification, molecular replacement in crystallography에서 개선을 보였다 (Extended Data Figs. 6-8 참조).

저자들은 "we hope that the methods we have described can be developed further and applied to benefit all areas of protein science"라며 향후 발전 방향을 제시했다. 이는 2년 후 AlphaFold 2로 이어진다.

## Limitations

1. **MSA 의존성**: 유사 서열이 적은 단백질(orphan protein)에서는 MSA quality가 떨어져 정확도가 급격히 감소한다.
2. **단일 도메인 제한**: Multi-domain protein의 domain 간 상대적 배치를 정확히 예측하지 못한다. 각 domain을 독립적으로 예측한 후 조합하는 방식의 한계가 있다.
3. **Gradient descent 최적화의 local minima**: L-BFGS로 에너지 landscape를 탐색하므로, 초기값에 따라 local minimum에 빠질 수 있다. 여러 random seed로 반복 최적화가 필요하다.
4. **Distogram 해상도 한계**: 64 bin으로 이산화된 거리 분포는 미세한 원자 간 거리 차이를 포착하기 어렵고, backbone torsion angle만 예측하므로 side-chain 배치가 부정확하다.
5. **End-to-end가 아님**: Feature extraction → distance prediction → structure optimization이 분리되어 있어, 전체 파이프라인의 joint optimization이 불가능하다.

## Conclusion

AlphaFold 1은 단백질 구조 예측의 패러다임을 fragment assembly에서 distance distribution prediction으로 전환시킨 획기적인 연구다. Distogram이라는 풍부한 inter-residue distance distribution 표현과, 이를 differentiable한 potential로 변환하여 gradient descent로 구조를 최적화하는 접근법은 CASP13에서 1위를 차지했다. Deep ResNet 기반의 distance prediction과 torsion prediction의 조합은 이후 AlphaFold 2의 end-to-end 구조 예측으로 가는 핵심 발판이 되었다.

## TL;DR

- **문제**: Fragment assembly는 느리고, contact prediction은 binary 정보만 제공하여 정확한 구조 예측이 어려움
- **해법**: Deep ResNet (220 blocks)으로 inter-residue distance distribution (distogram)을 예측하고, 이로부터 단백질별 potential을 구성하여 gradient descent로 구조 최적화
- **결과**: CASP13 FM category에서 압도적 1위 (52.8 vs 36.6 z-score), 이전에 불가능했던 새로운 fold들을 높은 정확도로 예측

## Paper Info

| 항목 | 내용 |
|------|------|
| **Title** | Improved protein structure prediction using potentials from deep learning |
| **Authors** | Andrew W. Senior et al. (DeepMind) |
| **Venue** | Nature, Volume 577 (2020) |
| **Published** | 2020-01-15 |
| **Link** | [doi:10.1038/s41586-019-1923-7](https://doi.org/10.1038/s41586-019-1923-7) |
| **Paper** | [Nature](https://www.nature.com/articles/s41586-019-1923-7) |
| **Code** | [GitHub](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13) |

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다. 
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
