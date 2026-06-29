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

Computational biology에서는 $c$가 protein sequence, binding pocket, target property, text instruction, scaffold, partial structure일 수 있습니다.

## Route Map

| Route | Use for | Start |
| --- | --- | --- |
| Distribution modeling | 어떤 distribution을 표현하고 sample을 어떻게 만드는가 | [Generative models](/concepts/generative-models), [Conditional generation](/concepts/generative-models/conditional-generation), [Sampling](/concepts/generative-models/sampling) |
| Likelihood and latent variables | explicit likelihood, encoder, decoder, lower bound | [Latent variable model](/concepts/generative-models/latent-variable-model), [ELBO](/concepts/generative-models/elbo), [VAE](/concepts/generative-models/vae) |
| Sequential generation | token, sequence, graph, action을 step by step 생성 | [Autoregressive model](/concepts/generative-models/autoregressive-model) |
| Denoising and score models | iterative corruption/reconstruction, score estimation, probability-flow view | [Diffusion model](/concepts/generative-models/diffusion-model), [Score-based model](/concepts/generative-models/score-based-model), [Probability flow ODE](/concepts/generative-models/probability-flow-ode) |
| Flow and velocity models | vector field, rectified path, invertible transformation | [Flow matching](/concepts/generative-models/flow-matching), [Rectified flow](/concepts/generative-models/rectified-flow), [Normalizing flow](/concepts/generative-models/normalizing-flow) |
| Energy and adversarial models | compatibility score, unnormalized density, generator-discriminator training | [Energy-based model](/concepts/generative-models/energy-based-model), [GAN](/concepts/generative-models/gan) |
| Control and speed | guidance, conditioning strength, few-step generation, sampler change | [Guidance](/concepts/generative-models/guidance), [Consistency model](/concepts/generative-models/consistency-model) |

## 모델 계열 구분

| Family | Learns | Typical sampling | Read |
| --- | --- | --- | --- |
| Autoregressive | next-token 또는 next-step conditional distribution | sequential decoding | [Autoregressive model](/concepts/generative-models/autoregressive-model) |
| Latent variable | latent representation과 decoder likelihood | latent를 sample한 뒤 output decode | [Latent variable model](/concepts/generative-models/latent-variable-model), [VAE](/concepts/generative-models/vae) |
| Adversarial | discriminator를 속이는 generator | direct generator pass | [GAN](/concepts/generative-models/gan) |
| Diffusion / score | denoising score 또는 noise prediction | iterative denoising | [Diffusion model](/concepts/generative-models/diffusion-model), [Score-based model](/concepts/generative-models/score-based-model) |
| Flow matching | probability path 위 vector field | ODE-like path 적분 | [Flow matching](/concepts/generative-models/flow-matching), [Rectified flow](/concepts/generative-models/rectified-flow) |
| Normalizing flow | invertible change of variables | base noise를 sample하고 map을 invert | [Normalizing flow](/concepts/generative-models/normalizing-flow) |
| Energy-based | unnormalized energy 또는 compatibility | MCMC, Langevin, optimization | [Energy-based model](/concepts/generative-models/energy-based-model) |
| Consistency | noisy state와 clean state 사이 direct jump | few-step 또는 one-step generation | [Consistency model](/concepts/generative-models/consistency-model) |

## Objective 기준

Likelihood-based model은 보통 negative log-likelihood를 optimize합니다.

$$
\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
[-\log p_\theta(x)]
$$

Latent-variable model은 unobserved variable $z$를 도입하고 tractable lower bound를 optimize합니다.

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
$$

Score/flow model은 direct likelihood를 main training target으로 쓰지 않는 경우가 많습니다. 대신 score, noise, denoised sample, velocity 같은 vector-valued target을 학습합니다.

$$
s_\theta(x_t,t) \approx \nabla_{x_t}\log p_t(x_t),
\qquad
v_\theta(x_t,t) \approx u_t(x_t)
$$

Paper를 읽을 때는 architecture를 비교하기 전에 무엇을 학습하는지부터 식별합니다.

## Claim Boundary

| Claim | 먼저 확인할 것 |
| --- | --- |
| better diffusion model | prediction target, sampler, NFE, guidance scale, evaluation axes |
| better flow model | probability path, time sampling, velocity target, solver budget |
| better conditional generation | condition source, condition leakage, fidelity-diversity tradeoff |
| better molecular/protein generation | validity, novelty, diversity, constraint satisfaction, downstream utility |
| faster sampling | matched quality at matched hardware, NFE, memory, and filtering rule |

## Generation Note Template

생성 모델 노트는 아래 항목이 있어야 비교가 됩니다.

| Field | Write |
| --- | --- |
| Sample space | text, sequence, graph, molecule, coordinate, image, action |
| Conditioning | class, text, pocket, scaffold, sequence, partial structure |
| Learned target | likelihood, next token, denoising noise, score, velocity, energy |
| Sampler | ancestral decoding, MCMC, ODE solver, denoising steps, direct generator |
| Control | guidance, constraints, rejection, filtering, reranking |
| Evaluation | quality, diversity, novelty, validity, utility, cost |

## Computational Biology 연결

| Context | Start |
| --- | --- |
| Molecules | [Molecular generation](/concepts/generative-models/molecular-generation), [Molecules](/molecular-modeling/molecules) |
| Proteins | [Protein design](/concepts/generative-models/protein-design), [Proteins](/molecular-modeling/proteins) |
| Structure-conditioned generation | [Structure-based modeling](/molecular-modeling/structure-based), [Geometry](/molecular-modeling/geometry) |

## 읽을 때 볼 질문

- 모델이 likelihood, score, velocity, denoising target 중 무엇을 학습하는가?
- conditioning 정보는 text, sequence, graph, structure 중 어디에서 오는가?
- sampling 과정에서 guidance, filtering, rejection이 sample distribution을 바꾸는가?
- validity, diversity, novelty, controllability를 어떻게 평가하는가?
- sample quality와 task utility를 분리해서 평가했는가?
