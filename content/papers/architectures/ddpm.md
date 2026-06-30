---
title: Denoising Diffusion Probabilistic Models
aliases:
  - papers/ddpm
  - papers/denoising-diffusion-probabilistic-models
  - papers/generative-models/ddpm
tags:
  - papers
  - architectures
  - generative-models
  - diffusion-model
  - image-generation
---

# Denoising Diffusion Probabilistic Models

> The paper made diffusion models a practical high-quality generative modeling recipe by training a neural network to reverse a fixed gradual noising process.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Denoising Diffusion Probabilistic Models |
| Authors | Jonathan Ho, Ajay Jain, Pieter Abbeel |
| Year | 2020 |
| Venue | NeurIPS 2020 |
| arXiv | [2006.11239](https://arxiv.org/abs/2006.11239) |
| Project page | [hojonathanho.github.io/diffusion](https://hojonathanho.github.io/diffusion/) |
| Code | [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) |
| Status | full note started |

## One-Line Takeaway

DDPM turns generation into iterative denoising: corrupt data with a known Gaussian Markov chain, train a neural network to reverse each step, and sample by starting from noise and repeatedly denoising.

## Question

Before diffusion became dominant, image generation was often read through:

- GANs for sharp samples;
- VAEs for likelihood-based latent-variable modeling;
- autoregressive models for tractable likelihood and sequential decoding;
- normalizing flows for exact invertible density models.

DDPM asks:

> Can a latent-variable Markov chain trained with a variational objective produce high-quality samples by reversing a simple noising process?

The generative contract:

$$
x_T \sim \mathcal{N}(0,I),
\qquad
x_{T-1},\ldots,x_0
\sim
p_\theta(x_{t-1}\mid x_t).
$$

The sample is $x_0$.

## Main Claim

The narrowed architecture/objective claim:

$$
\text{fixed Gaussian noising}
+
\text{learned denoising network}
+
\text{weighted variational objective}
\Rightarrow
\text{high-quality image synthesis}.
$$

The paper is important because it made a clean recipe:

1. define a forward diffusion process $q$ that gradually destroys data;
2. train a reverse process $p_\theta$ to undo it;
3. use a simplified denoising objective;
4. sample through many reverse steps.

## Architecture Contract

| Component | Role |
| --- | --- |
| data $x_0$ | clean image or sample |
| forward process $q$ | fixed Gaussian noising chain |
| reverse process $p_\theta$ | learned denoising chain |
| denoising network | predicts reverse transition statistics or noise |
| timestep embedding | tells the network which noise level it is denoising |
| noise schedule | controls corruption strength over time |
| sampling loop | iteratively maps Gaussian noise to data |

The model is an architecture family plus an objective, not a single network block. In images, the denoising network is often a U-Net-like architecture, but the DDPM idea can be used with other denoisers and domains.

## Forward Diffusion

The forward process gradually adds Gaussian noise:

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right),
$$

where:

- $t\in\{1,\ldots,T\}$;
- $\beta_t$ is the variance schedule;
- small $\beta_t$ means a small noising step.

Define:

$$
\alpha_t = 1-\beta_t,
\qquad
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s.
$$

Because the forward chain is Gaussian, there is a closed-form marginal:

$$
q(x_t\mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right).
$$

This allows one-step sampling of $x_t$ from $x_0$:

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

This equation is the practical training workhorse.

## Reverse Process

The learned reverse process is:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

Sampling starts from:

$$
p(x_T)=\mathcal{N}(0,I).
$$

Then:

$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0.
$$

The model has to learn local denoising transitions, not a direct one-shot map:

$$
z \rightarrow x.
$$

That is the main generative architecture shift.

## Noise Prediction Parameterization

The paper shows a useful parameterization where the network predicts the noise $\epsilon$ used to corrupt $x_0$:

$$
\epsilon_\theta(x_t,t)
\approx
\epsilon.
$$

Given:

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
$$

we can estimate:

$$
\hat{x}_0
=
\frac{
x_t
-
\sqrt{1-\bar{\alpha}_t}
\epsilon_\theta(x_t,t)
}
{\sqrt{\bar{\alpha}_t}}.
$$

The learned mean can be written in terms of predicted noise:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right).
$$

This form is why noise prediction became a standard diffusion interface.

## Simplified Training Objective

The simplified DDPM loss is:

$$
\mathcal{L}_{\text{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
\right)
\right\|_2^2
\right].
$$

Reading:

1. sample a real data point $x_0$;
2. sample a timestep $t$;
3. sample Gaussian noise $\epsilon$;
4. construct noisy $x_t$;
5. train the network to recover $\epsilon$.

The model is not trained by running the full chain during every training example. The closed-form noising equation makes training efficient.

## Variational Bound Reading

DDPM is a latent-variable model:

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
$$

The forward process defines an approximate posterior:

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
$$

The evidence lower bound can be written as:

$$
\log p_\theta(x_0)
\geq
\mathbb{E}_{q}
\left[
\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

The DDPM contribution is partly that this variational view connects to denoising score matching and yields a simple noise-prediction loss that works well in practice.

## Score Matching Connection

The score of a noisy distribution is:

$$
\nabla_{x_t}\log q(x_t).
$$

Under Gaussian corruption, predicting noise is related to predicting the score:

$$
s_\theta(x_t,t)
\approx
-
\frac{
\epsilon_\theta(x_t,t)
}{
\sqrt{1-\bar{\alpha}_t}
}.
$$

This connects DDPM to score-based generative modeling:

$$
\text{denoise}
\leftrightarrow
\text{estimate score}
\leftrightarrow
\text{reverse stochastic dynamics}.
$$

The paper is therefore a bridge between variational latent-variable modeling and score-based sampling.

## Sampling Algorithm

A simplified DDPM sampler:

1. sample $x_T\sim\mathcal{N}(0,I)$;
2. for $t=T,\ldots,1$:
3. predict $\epsilon_\theta(x_t,t)$;
4. compute $\mu_\theta(x_t,t)$;
5. sample $x_{t-1}$ from the reverse Gaussian;
6. return $x_0$.

Mathematically:

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_t z,
\qquad
z\sim\mathcal{N}(0,I).
$$

For $t=1$, implementations may omit extra noise.

This iterative sampler is the source of both:

- high sample quality;
- slow sampling relative to one-shot generators.

## Progressive Denoising

DDPM generation can be viewed as coarse-to-fine reconstruction:

$$
\text{noise}
\rightarrow
\text{coarse structure}
\rightarrow
\text{details}
\rightarrow
\text{sample}.
$$

The paper also frames the process as progressive lossy decompression. Early reverse steps recover broad semantic structure; later steps recover fine detail.

For later architectures, this view becomes important:

- image diffusion uses U-Net multi-scale denoisers;
- latent diffusion denoises compressed latent variables;
- video diffusion extends denoising over time;
- molecular diffusion denoises coordinates, graphs, or atom types.

## Relation to GANs

| Axis | GAN | DDPM |
| --- | --- | --- |
| generator | one-shot map from latent to sample | iterative denoising chain |
| training signal | adversarial discriminator | denoising/variational objective |
| likelihood | usually not tractable | variational bound available |
| sampling speed | fast | often slow |
| sample quality | sharp but unstable training risk | high quality with stable denoising training |
| mode coverage | can suffer mode collapse | generally better coverage behavior |

DDPM did not make GANs irrelevant immediately. It changed the tradeoff: slower sampling, but stable training and strong sample quality.

## Relation to VAEs

DDPM is a latent-variable model, but its latent variables are a long noising trajectory:

$$
x_1,\ldots,x_T.
$$

VAE-style model:

$$
z \sim p(z),
\qquad
x\sim p_\theta(x\mid z).
$$

DDPM-style model:

$$
x_T \sim \mathcal{N}(0,I),
\qquad
x_{t-1}\sim p_\theta(x_{t-1}\mid x_t).
$$

The decoder is not one big conditional distribution. It is many small denoising steps.

## Relation to Autoregressive Models

Autoregressive models factorize data by order:

$$
p(x)
=
\prod_i
p(x_i\mid x_{<i}).
$$

DDPM factorizes generation by noise level:

$$
p_\theta(x_0)
=
\int
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t)
dx_{1:T}.
$$

The decoding order is not pixel-by-pixel or token-by-token. It is denoising-time order.

## Relation to Flow Matching

DDPM learns stochastic reverse transitions:

$$
x_t
\rightarrow
x_{t-1}.
$$

Flow matching and rectified flow often learn a velocity field:

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t).
$$

Both are ways to transform a simple base distribution into a data distribution:

$$
\mathcal{N}(0,I)
\rightarrow
p_{\text{data}}.
$$

DDPM is the historical anchor for reading later diffusion/flow papers. Always ask:

- Is the process stochastic or deterministic?
- Is the target noise, score, clean data, or velocity?
- Is the sampler SDE-like, ODE-like, or discrete Markov?
- How many steps are used at evaluation?

## Relation to AlphaFold3 and Molecular Models

AlphaFold3 uses a diffusion-based architecture for biomolecular coordinate generation. The object differs, but the generative idea is recognizable:

$$
\text{noisy coordinates}
+
\text{conditioning context}
\rightarrow
\text{less noisy coordinates}.
$$

For molecular/protein models, DDPM provides the general reading vocabulary:

| DDPM Concept | Molecular/Structure Analogue |
| --- | --- |
| $x_0$ image | molecule, conformation, protein complex, coordinates |
| Gaussian noising | coordinate noise, graph corruption, categorical noise |
| denoising network | equivariant GNN, Transformer, structure module |
| timestep | noise level |
| condition | pocket, sequence, scaffold, text, target |
| sample validity | chemical/geometric validity |

This is why DDPM belongs in architecture notes, not only generative-model notes.

## Evidence Reading

The paper reports high-quality image synthesis, including strong CIFAR-10 FID and LSUN sample quality.

Evidence should be read by claim:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| CIFAR-10 FID | strong sample quality for small images | universal superiority across domains |
| LSUN samples | scalability to higher-resolution image domains | fast sampling |
| variational bound | likelihood-oriented interpretation | best likelihood among all models |
| rate-distortion curves | progressive compression interpretation | semantic controllability |
| sample visual quality | practical generative usefulness | calibrated probability or downstream utility |

The central evidence is that diffusion models can be competitive high-quality generators, not just theoretical latent-variable models.

## Implementation Notes

Implementation details that change behavior:

| Choice | Why It Matters |
| --- | --- |
| noise schedule $\beta_t$ | controls signal-to-noise progression |
| timestep embedding | lets denoiser condition on noise level |
| denoiser architecture | controls inductive bias and capacity |
| parameterization | noise, clean data, score, or velocity target |
| variance handling | affects likelihood and sampling |
| number of sampling steps | quality/latency tradeoff |
| data scaling | Gaussian assumptions depend on normalized data |

For images, the denoiser is often U-Net-like:

$$
\epsilon_\theta(x_t,t)
=
\operatorname{UNet}_\theta(x_t,\operatorname{embed}(t)).
$$

For graphs or 3D structures, the denoiser should match the object symmetry and representation.

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| ignoring sampler step count | quality and latency depend heavily on it |
| comparing FID without sampling budget | unfair if one model uses many more evaluations |
| treating diffusion as one architecture | denoiser architecture still matters |
| confusing noise prediction with score prediction conventions | scale and sign depend on parameterization |
| assuming likelihood and sample quality always align | they can trade off |
| porting image DDPM directly to molecules | chemical validity and symmetry need domain-specific design |
| ignoring conditioning leakage | conditional generation can use unavailable information |

## Common Misreadings

### "DDPM is just adding noise and removing it."

That is the intuition, but the paper's value is the probabilistic formulation, the closed-form noising marginal, the reverse Markov chain, and the simplified denoising objective.

### "The denoising network is the whole architecture."

No. The architecture includes the noising process, timestep parameterization, reverse transition parameterization, denoising network, and sampler.

### "Diffusion models are always slow."

Original DDPM sampling is slow because it uses many denoising steps. Later samplers and parameterizations reduce the step count, but the speed-quality tradeoff remains a core reading axis.

### "DDPM only matters for images."

No. The image results made it visible, but the diffusion recipe became a general template for audio, video, 3D, molecules, proteins, and control.

## Later-Paper Checklist

When reading later diffusion or flow papers, check:

- What is the corrupted object: pixels, latents, tokens, graphs, coordinates, fields?
- Is the process discrete-time or continuous-time?
- Does the model predict noise, score, clean data, or velocity?
- What is the noising schedule?
- What denoiser architecture is used?
- Is the sampler stochastic or deterministic?
- How many sampling steps are reported?
- Is guidance used?
- Is evaluation budget matched?
- Are invalid samples filtered before metrics?
- Does the model preserve domain symmetries such as permutation, rotation, or translation?

## Why It Matters

DDPM is a foundation paper for modern generative modeling because it turned a conceptually simple denoising process into a competitive generative architecture.

For this wiki, it connects:

$$
\text{latent variable models}
\rightarrow
\text{score matching}
\rightarrow
\text{iterative denoising}
\rightarrow
\text{modern diffusion and flow models}.
$$

It also gives the vocabulary needed to read diffusion-based scientific architectures such as AlphaFold3, molecular conformation generation, and protein design models.

## Limitations

The original DDPM recipe has important limitations:

- sampling requires many steps;
- image denoisers do not automatically transfer to structured scientific objects;
- sample quality depends strongly on schedule and architecture;
- likelihood and perceptual quality are not identical;
- evaluation metrics such as FID are imperfect;
- conditional generation and guidance require separate analysis.

The defensible claim:

$$
\text{DDPM}
\Rightarrow
\text{diffusion as a practical high-quality generative model family}.
$$

The overclaim to avoid:

$$
\text{DDPM}
\Rightarrow
\text{all generative modeling problems are solved}.
$$

## Connections

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/elbo|ELBO]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[papers/architectures/u-net|U-Net]]
- [[papers/architectures/alphafold3|AlphaFold3]]
- [[papers/generative-models/index|Generative Model Papers]]
- [[papers/architectures/index|Architecture papers]]
