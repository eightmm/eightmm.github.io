---
title: Generative Models
aliases:
  - research/generative-models
  - research/generative-models/index
tags:
  - ai
  - generative-models
---

# Generative Models

생성 모델은 data distribution에서 sample을 만들거나, noise에서 structure를 복원하거나, 조건에 맞는 output을 구성하는 모델군입니다.

핵심 목표는 실제 데이터 분포 $p_{\mathrm{data}}(x)$를 잘 근사하는 모델 분포 $p_\theta(x)$를 만드는 것입니다.

$$
p_\theta(x) \approx p_{\mathrm{data}}(x)
$$

Conditional generation에서는 조건 $c$가 주어졌을 때의 분포를 모델링합니다.

$$
x \sim p_\theta(x \mid c)
$$

Bio-AI에서는 $c$가 protein sequence, binding pocket, target property, text instruction, scaffold, or partial structure일 수 있습니다.

## Core Notes

- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/consistency-model|Consistency model]]

## Bio-AI Links

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[ai/generative-models|Generative models]]

## Questions

- 모델이 likelihood, score, velocity, denoising target 중 무엇을 학습하는가?
- conditioning 정보는 text, sequence, graph, structure 중 어디에서 오는가?
- validity, diversity, novelty, controllability를 어떻게 평가하는가?
- sample quality와 task utility를 분리해서 평가했는가?
