---
title: Generative Models
tags:
  - generative-models
  - machine-learning
---

# Generative Models

Generative model notes describe ways to model and sample data distributions, including sequences, graphs, coordinates, molecules, and proteins.

The shared goal is to learn a model distribution close to the data distribution:

$$
p_\theta(x) \approx p_{\mathrm{data}}(x)
$$

Different families choose different training signals: likelihood, reconstruction, adversarial discrimination, denoising, score estimation, or velocity matching.

## Selection Axes

Choose a generative family by the object, output validity rule, likelihood need, sampling budget, and conditioning mechanism.

| Family | Training signal | Sampling shape | Strength | Common risk |
| --- | --- | --- | --- | --- |
| [Autoregressive](/concepts/generative-models/autoregressive-model) | Next-token likelihood | Sequential | Stable likelihood training | Slow long-horizon sampling |
| [VAE](/concepts/generative-models/vae) | ELBO | Latent decode | Structured latent space | Posterior collapse, blurry samples |
| [GAN](/concepts/generative-models/gan) | Adversarial game | One-shot generator | Sharp samples | Mode collapse, unstable training |
| [Normalizing flow](/concepts/generative-models/normalizing-flow) | Exact likelihood | Invertible transform | Tractable density | Invertibility constraints |
| [Diffusion](/concepts/generative-models/diffusion-model) | Denoising/noise prediction | Iterative denoising | Stable high-quality samples | Many sampling steps |
| [Score-based](/concepts/generative-models/score-based-model) | Score matching | SDE/ODE sampler | Continuous-time view | Noise-level coverage |
| [Flow matching](/concepts/generative-models/flow-matching) | Velocity matching | ODE transport | Direct path learning | Path and symmetry design |
| [Consistency](/concepts/generative-models/consistency-model) | Trajectory consistency | One/few step | Fast sampling | Distillation or consistency quality |

## Evaluation Boundary

Generation quality is not one metric. A useful evaluation names:

- Validity: does the sample obey the syntax, physics, geometry, or task constraints?
- Diversity: does the model cover modes rather than repeat a few outputs?
- Novelty: is the sample distinct from training examples under the right equivalence relation?
- Utility: does the generated object improve the downstream objective?
- Calibration or likelihood: if probabilities are used, are they meaningful for the decision?
- Cost: how many steps, tokens, calls, or device-hours are needed per useful sample?

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
