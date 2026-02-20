---
title: "MolCrystalFlow: Molecular Crystal Structure Prediction via Flow Matching"
date: 2026-02-19 13:00:00 +0900
categories: [AI, Paper Review]
tags: [flow-matching, crystal-structure, Riemannian, SE3-equivariant, rigid-body, GNN]
math: true
---

## 📌 요약

분자 결정 구조 예측(CSP)을 위한 Riemannian manifold 위의 flow matching 모델. 분자를 rigid body로 표현하고, lattice matrix, 분자 방향(orientation), 중심 위치(centroid)를 공동으로 학습한다.

> **중요도:** ⭐⭐⭐⭐ | **약어:** MolCrystalFlow

## 핵심 기여

1. **계층적 표현**: 분자 내부 복잡성과 분자 간 패킹을 분리. EGNN으로 E(3)-불변 임베딩 후 rigid body로 처리
2. **Riemannian Flow Matching**: Lattice는 선형 보간, centroid는 Torus $T^3$ geodesic, orientation은 $SO(3)$ geodesic flow
3. **Periodic E(3)-Invariant GNN**: 주기적 경계 조건과 SE(3) 대칭을 보존하는 메시지 패싱
4. **χ-grouped Optimal Transport**: Axis-flip state별 OT 그룹화로 cross-link 감소
5. **SOTA 성능**: CSD 데이터셋에서 lattice volume RMAD **3.86%** (MOFFlow 18.8%, Genarris-3 59.0%)

## 방법론

### 2단계 계층 모델

**Stage 1 — Building Block Embedder (EGNN):**
각 분자를 E(3)-불변 임베딩으로 변환. 보조 특징 18개(원자 수, chirality, logP, radius of gyration 등) 포함.

**Stage 2 — MolCrystalNet:**
각 modality를 고유한 Riemannian manifold에서 flow matching:

| Modality | Manifold | Interpolation |
|----------|----------|---------------|
| Lattice $L \in \mathbb{R}^{3\times3}$ | Euclidean | 선형 보간 |
| Centroid $F \in T^3$ | Torus | Geodesic (wrapping) |
| Orientation $R \in SO(3)$ | SO(3) | $R_t = R_0 \cdot \exp(t \cdot \log(R_0^T R_1))$ |

### Velocity Annealing

성능 향상을 위한 velocity scaling factor: $s_{uF} \in [5, 13]$, $s_{uR} \in [1, 3]$. 최적값 $s_{uF}=9, s_{uR}=3$에서 matching rate 3.36% → **6.8%** 향상.

## 실험 결과

- **Lattice volume RMAD**: 3.86% (MOFFlow 18.8%, Genarris-3 raw 59%)
- **생성 속도**: 22ms/구조 (Genarris-3 43ms 대비 2배)
- **CCDC Blind Test Target VIII**: PBE-MBD 최저 에너지 polymorph가 실험 구조와 유사 (RMSD₇ = 0.397 Å)

### CSP 파이프라인

1. MolCrystalFlow로 1000개 후보 생성
2. u-MLIP으로 2단계 relaxation
3. 에너지 기준 Top-10 선택
4. DFT (PBE-D3, PBE-MBD) 랭킹

## 한계 및 향후 연구

- 에너지 정보 미활용 (구조만 학습)
- Rigid body 가정 (conformational polymorphism 처리 불가)
- Space group 대칭 미활용

## 연구 연결점

SO(3) geodesic flow, axis-angle $(ω, κ, ρ)$ embedding, χ-grouped OT, velocity annealing — **PL 도킹의 SE(3) flow matching에 직접 참고 가능**.

## 링크

- 📄 [arXiv: 2602.16020](https://arxiv.org/abs/2602.16020)
