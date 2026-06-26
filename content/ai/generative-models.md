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

Molecular modeling에서는 $c$가 protein sequence, binding pocket, target property, text instruction, scaffold, or partial structure일 수 있습니다.

## 핵심 노트

- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/consistency-model|Consistency model]]

## 모델 계열 구분

| Family | Learns | Typical Sampling | Read |
| --- | --- | --- | --- |
| Autoregressive | next-token or next-step conditional distribution | sequential decoding | [Autoregressive model](/concepts/generative-models/autoregressive-model) |
| Latent variable | latent representation and decoder likelihood | sample latent, decode output | [Latent variable model](/concepts/generative-models/latent-variable-model), [VAE](/concepts/generative-models/vae) |
| Adversarial | generator that fools a discriminator | direct generator pass | [GAN](/concepts/generative-models/gan) |
| Diffusion / score | denoising score or noise prediction | iterative denoising | [Diffusion model](/concepts/generative-models/diffusion-model), [Score-based model](/concepts/generative-models/score-based-model) |
| Flow matching | vector field along a probability path | integrate an ODE-like path | [Flow matching](/concepts/generative-models/flow-matching), [Rectified flow](/concepts/generative-models/rectified-flow) |
| Normalizing flow | invertible change of variables | sample base noise, invert map | [Normalizing flow](/concepts/generative-models/normalizing-flow) |
| Consistency | direct jump between noisy and clean states | few-step or one-step generation | [Consistency model](/concepts/generative-models/consistency-model) |

## Objective 기준

Likelihood-based models usually optimize a negative log-likelihood:

$$
\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
[-\log p_\theta(x)]
$$

Latent-variable models introduce an unobserved variable $z$ and optimize a tractable lower bound:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
$$

Score and flow models often avoid direct likelihood as the main training target. They learn a vector-valued target such as a score, noise, denoised sample, or velocity:

$$
s_\theta(x_t,t) \approx \nabla_{x_t}\log p_t(x_t),
\qquad
v_\theta(x_t,t) \approx u_t(x_t)
$$

When reading a paper, first identify the learned quantity before comparing architectures.

## Molecular Modeling 연결

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[ai/generative-models|Generative models]]

## 읽을 때 볼 질문

- 모델이 likelihood, score, velocity, denoising target 중 무엇을 학습하는가?
- conditioning 정보는 text, sequence, graph, structure 중 어디에서 오는가?
- sampling 과정에서 guidance, filtering, rejection이 sample distribution을 바꾸는가?
- validity, diversity, novelty, controllability를 어떻게 평가하는가?
- sample quality와 task utility를 분리해서 평가했는가?
