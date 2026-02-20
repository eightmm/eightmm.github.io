---
title: "SpecLig: Energy-Guided Hierarchical Model for Target-Specific 3D Ligand Design"
date: 2026-02-19 15:00:00 +0900
categories: [AI, Drug Discovery]
tags: [protein-ligand, drug-design, diffusion, equivariant, specificity, VAE]
math: true
---

## 📌 요약

Hierarchical SE(3)-equivariant VAE + energy-guided latent diffusion으로 **친화도와 특이성을 동시에 달성**하는 리간드 생성 프레임워크.

> **중요도:** ⭐⭐⭐⭐⭐ | **약어:** SpecLig

## 핵심 아이디어

기존 structure-based drug design (SBDD) 모델은 target에 대한 높은 binding affinity는 달성하지만, **off-target selectivity (특이성)**을 무시하는 경우가 많다. SpecLig는 다음을 결합하여 이 문제를 해결한다:

1. **Hierarchical SE(3)-equivariant VAE**: 분자의 multi-scale 표현 학습
2. **Energy-guided latent diffusion**: 물리 기반 에너지 함수로 latent space에서의 생성을 가이드
3. **Target-specificity**: 특정 타겟에만 강하게 결합하고 off-target에는 약하게 결합하는 리간드 생성

## 연구 연결점

- SE(3)-equivariant architecture + diffusion의 결합
- Energy-guided generation은 flow matching에도 적용 가능한 패러다임
- Protein-ligand 연구에서 **specificity를 명시적으로 다루는** 드문 사례

## 링크

- 📄 [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.06.687093v1)
