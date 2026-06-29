---
title: Generative Models
tags:
  - generative-models
  - machine-learning
---

# Generative Models

Generative model note는 sequence, graph, coordinate, molecule, protein을 포함한 data distribution을 모델링하고 sampling하는 방법을 설명합니다.

공통 목표는 data distribution에 가까운 model distribution을 학습하는 것입니다.

$$
p_\theta(x) \approx p_{\mathrm{data}}(x)
$$

Different families choose different training signals: likelihood, reconstruction, adversarial discrimination, denoising, score estimation, or velocity matching.

## Generative Model Contract

A generative model should specify the sample space, the conditioning variable, the learned objective, the sampler, and the validation rule.

$$
\mathcal{G}
=
(\mathcal{X},\ c,\ p_\theta,\ \mathcal{L},\ \pi_{\mathrm{sample}},\ V)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $\mathcal{X}$ | output space | text, image, graph, molecule, protein sequence, 3D coordinate? |
| $c$ | conditioning context | class label, prompt, scaffold, pocket, sequence, partial structure? |
| $p_\theta$ or score field | model distribution or learned field | explicit density, implicit generator, energy, score, velocity? |
| $\mathcal{L}$ | training objective | likelihood, ELBO, denoising loss, score loss, adversarial loss, velocity loss? |
| $\pi_{\mathrm{sample}}$ | sampling procedure | autoregressive decode, ODE solver, denoising chain, MCMC, direct generator? |
| $V$ | validity rule | syntax, chemistry, geometry, constraints, downstream test? |

Conditional generation uses a distribution over outputs given context:

$$
x \sim p_\theta(x \mid c)
$$

For scientific objects, $V(x,c)$ is often as important as $p_\theta(x\mid c)$ because invalid samples are not merely low quality; they may be outside the intended sample space.

## Selection Axes

Choose a generative family by the object, output validity rule, likelihood need, sampling budget, and conditioning mechanism.

| Family | Training signal | Sampling shape | Strength | Common risk |
| --- | --- | --- | --- | --- |
| [Autoregressive](/concepts/generative-models/autoregressive-model) | Next-token likelihood | Sequential | Stable likelihood training | Slow long-horizon sampling |
| [VAE](/concepts/generative-models/vae) | ELBO | Latent decode | Structured latent space | Posterior collapse, blurry samples |
| [GAN](/concepts/generative-models/gan) | Adversarial game | One-shot generator | Sharp samples | Mode collapse, unstable training |
| [Normalizing flow](/concepts/generative-models/normalizing-flow) | Exact likelihood | Invertible transform | Tractable density | Invertibility constraints |
| [Energy-based model](/concepts/generative-models/energy-based-model) | Energy or contrastive objective | MCMC/Langevin/optimization | Flexible unnormalized density | Partition function and sampling cost |
| [Diffusion](/concepts/generative-models/diffusion-model) | Denoising/noise prediction | Iterative denoising | Stable high-quality samples | Many sampling steps |
| [Score-based](/concepts/generative-models/score-based-model) | Score matching | SDE/ODE sampler | Continuous-time view | Noise-level coverage |
| [Flow matching](/concepts/generative-models/flow-matching) | Velocity matching | ODE transport | Direct path learning | Path and symmetry design |
| [Consistency](/concepts/generative-models/consistency-model) | Trajectory consistency | One/few step | Fast sampling | Distillation or consistency quality |

## Objective, Sampler, Evaluation

Do not collapse these three layers. Two papers can use the same architecture while changing only the training objective or sampler.

$$
\text{model family}
\neq
\text{training objective}
\neq
\text{sampling algorithm}
\neq
\text{evaluation protocol}
$$

| Layer | Examples | What can go wrong |
| --- | --- | --- |
| Objective | NLL, ELBO, denoising MSE, score matching, flow matching, adversarial loss | objective improves but samples do not |
| Parameterization | noise, clean sample, score, velocity, energy, logits | papers compare different targets as if they were architectures |
| Sampler | ancestral sampling, beam search, DDIM, SDE/ODE solver, MCMC, rejection, reranking | quality changes because sampling budget changed |
| Conditioning | prompt, label, structure, scaffold, pocket, sequence, retrieval | condition leakage or unfair context |
| Evaluation | validity, novelty, diversity, utility, likelihood, cost | filtered samples hide failure rate |

The denominator matters:

$$
\text{success rate}
=
\frac{\#\text{valid and useful samples}}
{\#\text{attempted samples}}
$$

Reporting only kept samples changes the claim:

$$
\frac{\#\text{useful kept samples}}{\#\text{kept samples}}
\neq
\frac{\#\text{useful samples}}{\#\text{generated samples}}
\neq
\frac{\#\text{useful samples}}{\#\text{sampling attempts}}
$$

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| do you need exact likelihood? | [Autoregressive model](/concepts/generative-models/autoregressive-model), [Normalizing flow](/concepts/generative-models/normalizing-flow) | factorization, ordering, invertibility |
| is density only defined up to normalization? | [Energy-based model](/concepts/generative-models/energy-based-model), [Score matching](/concepts/generative-models/score-matching) | partition function, negative samples, sampler |
| is there a latent variable? | [Latent variable model](/concepts/generative-models/latent-variable-model), [ELBO](/concepts/generative-models/elbo), [VAE](/concepts/generative-models/vae) | reconstruction vs prior sampling |
| is generation iterative? | [Diffusion model](/concepts/generative-models/diffusion-model), [Score matching](/concepts/generative-models/score-matching), [Flow matching](/concepts/generative-models/flow-matching) | sampler, step count, parameterization |
| is generation conditional? | [Conditional generation](/concepts/generative-models/conditional-generation), [Guidance](/concepts/generative-models/guidance) | condition leakage and guidance strength |
| is the object molecular or structural? | [Molecular generation](/concepts/generative-models/molecular-generation), [Protein design](/concepts/generative-models/protein-design) | validity, novelty, split, physical plausibility |
| how should samples be judged? | [Generation evaluation](/concepts/evaluation/generation-evaluation), [Sampling](/concepts/generative-models/sampling) | attempted vs kept samples |

## Generation Route Map

| Route | Core object | Main equation | Best first read |
| --- | --- | --- | --- |
| Likelihood | normalized density | $\max_\theta \log p_\theta(x)$ | [[concepts/math/maximum-likelihood|Maximum likelihood]] |
| Autoregressive | factorized sequence density | $p_\theta(x)=\prod_t p_\theta(x_t\mid x_{<t})$ | [[concepts/generative-models/autoregressive-model|Autoregressive model]] |
| Latent variable | latent-conditioned density | $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$ | [[concepts/generative-models/latent-variable-model|Latent variable model]] |
| VAE / ELBO | variational lower bound | $\log p_\theta(x)\ge \mathcal{L}_{\mathrm{ELBO}}$ | [[concepts/generative-models/elbo|Evidence lower bound]] |
| Energy-based | unnormalized density | $p_\theta(x)=\exp(-E_\theta(x))/Z_\theta$ | [[concepts/generative-models/energy-based-model|Energy-based model]] |
| Diffusion / score | denoising or score field | $s_\theta(x_t,t)\approx \nabla_{x_t}\log p_t(x_t)$ | [[concepts/generative-models/diffusion-model|Diffusion model]] |
| Flow matching | probability-path velocity | $\frac{dx_t}{dt}=v_\theta(x_t,t)$ | [[concepts/generative-models/flow-matching|Flow matching]] |
| Normalizing flow | invertible transformation | $\log p_X(x)=\log p_Z(f^{-1}(x))+\log|\det J_{f^{-1}}|$ | [[concepts/generative-models/normalizing-flow|Normalizing flow]] |
| Consistency | trajectory-consistent map | $f_\theta(x_t,t)\approx f_\theta(x_s,s)$ | [[concepts/generative-models/consistency-model|Consistency model]] |

## Evaluation Boundary

Generation quality는 하나의 metric으로 끝나지 않습니다. 유용한 evaluation은 아래 항목을 명시합니다.

- Validity: does the sample obey the syntax, physics, geometry, or task constraints?
- Diversity: does the model cover modes rather than repeat a few outputs?
- Novelty: is the sample distinct from training examples under the right equivalence relation?
- Utility: does the generated object improve the downstream objective?
- Calibration or likelihood: if probabilities are used, are they meaningful for the decision?
- Cost: how many steps, tokens, calls, or device-hours are needed per useful sample?

## Domain Boundaries

| Domain | Keep in generative models | Move to |
| --- | --- | --- |
| text or sequence generation | factorization, decoding, likelihood, sampling budget | [[ai/architectures|Architectures]], [[agents/index|Agents]] when tool use is involved |
| molecule generation | objective, sampler, validity denominator | [[molecular-modeling/molecules|Molecules]], [[molecular-modeling/data-evaluation|Molecular modeling data and evaluation]] |
| protein design | sequence/structure-conditioned generation contract | [[molecular-modeling/proteins|Proteins]], [[molecular-modeling/structure-based/index|Structure-based modeling]] |
| 3D structure generation | equivariance, coordinate validity, geometry constraints | [[concepts/geometric-deep-learning/index|Geometric deep learning]], [[molecular-modeling/geometry-for-structure-modeling|Geometry for Structure Modeling]] |
| serving generated outputs | latency, batching, cost, reproducibility | [[ai/systems|AI Systems]], [[infra/index|Infra]] |

## Common Traps

| Trap | Why it matters |
| --- | --- |
| Reporting only filtered samples | hides invalid generation rate and sampler cost |
| Changing NFE or decode budget | makes speed and quality comparisons unfair |
| Mixing novelty with utility | new samples can be useless, useful samples can be close analogs |
| Treating likelihood as sample quality | high likelihood does not guarantee useful samples |
| Treating sample quality as calibration | pretty or valid samples do not imply meaningful probabilities |
| Ignoring constraint repair | post-processing can become part of the generator |
| Evaluating against leaked training neighbors | novelty and generalization claims become weak |

## Common Concepts

- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/elbo|Evidence lower bound]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]

## Core Families

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/energy-based-model|Energy-based model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/consistency-model|Consistency model]]

## Scientific Targets

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[entities/molecule|Molecule]]
- [[entities/protein|Protein]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[ai/generative-models|Generative models]]
