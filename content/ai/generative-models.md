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

Computational biology에서는 $c$가 protein sequence, binding pocket, target property, text instruction, scaffold, or partial structure일 수 있습니다.

## Route Map

| Route | Use For | Start |
| --- | --- | --- |
| Distribution modeling | what distribution is represented and how samples are produced | [Generative models](/concepts/generative-models), [Conditional generation](/concepts/generative-models/conditional-generation), [Sampling](/concepts/generative-models/sampling) |
| Likelihood and latent variables | explicit likelihoods, encoders, decoders, lower bounds | [Latent variable model](/concepts/generative-models/latent-variable-model), [ELBO](/concepts/generative-models/elbo), [VAE](/concepts/generative-models/vae) |
| Sequential generation | token, sequence, graph, or action generation step by step | [Autoregressive model](/concepts/generative-models/autoregressive-model) |
| Denoising and score models | iterative corruption/reconstruction, score estimation, probability-flow view | [Diffusion model](/concepts/generative-models/diffusion-model), [Score-based model](/concepts/generative-models/score-based-model), [Probability flow ODE](/concepts/generative-models/probability-flow-ode) |
| Flow and velocity models | vector fields, rectified paths, invertible transformations | [Flow matching](/concepts/generative-models/flow-matching), [Rectified flow](/concepts/generative-models/rectified-flow), [Normalizing flow](/concepts/generative-models/normalizing-flow) |
| Energy and adversarial models | compatibility scores, unnormalized densities, generator-discriminator training | [Energy-based model](/concepts/generative-models/energy-based-model), [GAN](/concepts/generative-models/gan) |
| Control and speed | guidance, conditioning strength, few-step generation, sampler changes | [Guidance](/concepts/generative-models/guidance), [Consistency model](/concepts/generative-models/consistency-model) |

## 모델 계열 구분

| Family | Learns | Typical Sampling | Read |
| --- | --- | --- | --- |
| Autoregressive | next-token or next-step conditional distribution | sequential decoding | [Autoregressive model](/concepts/generative-models/autoregressive-model) |
| Latent variable | latent representation and decoder likelihood | sample latent, decode output | [Latent variable model](/concepts/generative-models/latent-variable-model), [VAE](/concepts/generative-models/vae) |
| Adversarial | generator that fools a discriminator | direct generator pass | [GAN](/concepts/generative-models/gan) |
| Diffusion / score | denoising score or noise prediction | iterative denoising | [Diffusion model](/concepts/generative-models/diffusion-model), [Score-based model](/concepts/generative-models/score-based-model) |
| Flow matching | vector field along a probability path | integrate an ODE-like path | [Flow matching](/concepts/generative-models/flow-matching), [Rectified flow](/concepts/generative-models/rectified-flow) |
| Normalizing flow | invertible change of variables | sample base noise, invert map | [Normalizing flow](/concepts/generative-models/normalizing-flow) |
| Energy-based | unnormalized energy or compatibility | MCMC, Langevin, or optimization | [Energy-based model](/concepts/generative-models/energy-based-model) |
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

## Claim Boundary

| Claim | First Check |
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
