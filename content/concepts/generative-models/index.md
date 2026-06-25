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

## Core Families

- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/consistency-model|Consistency model]]

## Scientific Targets

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]

## Related

- [[entities/molecule|Molecule]]
- [[entities/protein|Protein]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[ai/generative-models|Generative models]]
