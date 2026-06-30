---
title: Auto-Encoding Variational Bayes
aliases:
  - papers/vae
  - papers/auto-encoding-variational-bayes
  - papers/generative-models/vae
tags:
  - papers
  - architectures
  - generative-models
  - vae
  - latent-variable-model
---

# Auto-Encoding Variational Bayes

> The paper introduced the variational autoencoder recipe: learn a neural latent-variable generative model with an amortized inference network and the reparameterization trick.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Auto-Encoding Variational Bayes |
| Authors | Diederik P. Kingma, Max Welling |
| Year | 2013 preprint; 2014 conference |
| Venue | ICLR 2014 |
| arXiv | [1312.6114](https://arxiv.org/abs/1312.6114) |
| OpenReview | [33X9fd2-9FyZd](https://openreview.net/forum?id=33X9fd2-9FyZd) |
| Status | full note started |

## One-Line Takeaway

VAE turns latent-variable modeling into a trainable neural architecture by pairing a decoder $p_\theta(x\mid z)$ with an encoder $q_\phi(z\mid x)$ and optimizing an evidence lower bound using differentiable stochastic sampling.

## Question

Latent-variable generative models assume observed data $x$ is generated through hidden variables $z$:

$$
z \sim p(z),
\qquad
x \sim p_\theta(x\mid z).
$$

The marginal likelihood is:

$$
p_\theta(x)
=
\int
p_\theta(x,z)\,dz
=
\int
p_\theta(x\mid z)p(z)\,dz.
$$

The problem:

$$
\log p_\theta(x)
$$

is often hard to compute because the integral over $z$ is intractable.

The paper asks:

> Can we train deep latent-variable models with continuous latent variables using stochastic gradients, even when exact posterior inference is intractable?

## Main Claim

The narrowed claim:

$$
\text{amortized inference network}
+
\text{reparameterization trick}
+
\text{variational lower bound}
\Rightarrow
\text{scalable neural latent-variable generative models}.
$$

The architecture pattern:

$$
x
\xrightarrow{\text{encoder }q_\phi(z\mid x)}
z
\xrightarrow{\text{decoder }p_\theta(x\mid z)}
\hat{x}.
$$

The generative sampling path:

$$
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
$$

Those are different paths. Reconstruction quality and prior-sample quality must be read separately.

## Architecture Contract

| Component | Role |
| --- | --- |
| prior $p(z)$ | simple latent distribution, often Gaussian |
| decoder $p_\theta(x\mid z)$ | generative model from latent to observation |
| encoder $q_\phi(z\mid x)$ | approximate posterior / recognition model |
| ELBO | tractable lower bound on log likelihood |
| reparameterization trick | lets gradients pass through stochastic latent samples |
| reconstruction term | encourages decoder to explain $x$ |
| KL term | regularizes approximate posterior toward prior |

The VAE is not merely an autoencoder with noise. It is a probabilistic latent-variable model with an inference network.

## Generative Model

The joint distribution is:

$$
p_\theta(x,z)
=
p_\theta(x\mid z)p(z).
$$

A common prior is:

$$
p(z)
=
\mathcal{N}(0,I).
$$

The decoder defines:

$$
p_\theta(x\mid z).
$$

For continuous data, the decoder may define a Gaussian:

$$
p_\theta(x\mid z)
=
\mathcal{N}
\left(
x;
\mu_\theta(z),
\sigma^2 I
\right).
$$

For binary data, it may define Bernoulli probabilities:

$$
p_\theta(x\mid z)
=
\prod_i
\operatorname{Bernoulli}(x_i;\pi_{\theta,i}(z)).
$$

The decoder likelihood must match the data type. This choice is part of the architecture contract.

## Inference Model

The true posterior:

$$
p_\theta(z\mid x)
=
\frac{
p_\theta(x\mid z)p(z)
}{
p_\theta(x)
}
$$

is often intractable because $p_\theta(x)$ is intractable.

VAE introduces an approximate posterior:

$$
q_\phi(z\mid x).
$$

This is the encoder or recognition model. It maps an observed example to a distribution over latent variables:

$$
x
\rightarrow
(\mu_\phi(x),\sigma_\phi(x)).
$$

For a diagonal Gaussian encoder:

$$
q_\phi(z\mid x)
=
\mathcal{N}
\left(
z;
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

This is amortized inference: instead of optimizing a separate variational distribution for every datapoint, one shared neural network predicts the variational parameters.

## Evidence Lower Bound

Start from:

$$
\log p_\theta(x)
=
\log
\int
p_\theta(x,z)\,dz.
$$

Insert $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=
\log
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{
p_\theta(x,z)
}{
q_\phi(z\mid x)
}
\right].
$$

Apply Jensen's inequality:

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
$$

This is the ELBO:

$$
\mathcal{L}(\theta,\phi;x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\Vert
p(z)
\right).
$$

The two terms:

| Term | Formula | Meaning |
| --- | --- | --- |
| reconstruction / likelihood | $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$ | decoder explains data sampled from encoder posterior |
| KL regularization | $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))$ | approximate posterior stays close to prior |

## KL Gap

The ELBO is lower than the true log likelihood by:

$$
\log p_\theta(x)
-
\mathcal{L}(\theta,\phi;x)
=
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\Vert
p_\theta(z\mid x)
\right).
$$

The bound is tight when:

$$
q_\phi(z\mid x)
=
p_\theta(z\mid x).
$$

This makes posterior approximation an explicit part of the model. A weak encoder can lower the bound even if the decoder family is expressive.

## Reparameterization Trick

The encoder samples:

$$
z\sim q_\phi(z\mid x).
$$

Naively sampling $z$ blocks straightforward backpropagation through $\phi$. The reparameterization trick writes the random variable as a deterministic transformation of parameter-free noise:

$$
\epsilon\sim\mathcal{N}(0,I),
$$

$$
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon.
$$

Then:

$$
\nabla_\phi
\mathbb{E}_{q_\phi(z\mid x)}
[f(z)]
$$

can be estimated by:

$$
\nabla_\phi
\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}
\left[
f(
\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
)
\right].
$$

This is the key engineering move that made neural variational inference easy to optimize with standard stochastic gradient methods.

## Training Objective

Training usually minimizes negative ELBO:

$$
\mathcal{J}(\theta,\phi)
=
-
\mathbb{E}_{x\sim p_{\text{data}}}
\left[
\mathcal{L}(\theta,\phi;x)
\right].
$$

Equivalently:

$$
\mathcal{J}
=
\mathcal{L}_{\text{recon}}
+
\mathcal{L}_{\text{KL}},
$$

where sign conventions vary by implementation.

For a Gaussian decoder with fixed variance, reconstruction becomes proportional to squared error:

$$
-\log p_\theta(x\mid z)
\propto
\lVert x-\mu_\theta(z)\rVert_2^2.
$$

For a Bernoulli decoder, reconstruction becomes binary cross entropy:

$$
-\log p_\theta(x\mid z)
=
-
\sum_i
\left[
x_i\log \pi_i
+
(1-x_i)\log(1-\pi_i)
\right].
$$

Always check the decoder likelihood before interpreting reconstruction loss.

## Closed-Form KL for Gaussian Encoder

If:

$$
q_\phi(z\mid x)
=
\mathcal{N}(\mu,\operatorname{diag}(\sigma^2)),
\qquad
p(z)=\mathcal{N}(0,I),
$$

then:

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
=
\frac{1}{2}
\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
\right).
$$

This closed form is one reason the diagonal Gaussian VAE became the standard first implementation.

## Autoencoder vs VAE

| Axis | Autoencoder | VAE |
| --- | --- | --- |
| encoder output | point code $z=f(x)$ | distribution $q_\phi(z\mid x)$ |
| decoder output | reconstruction $\hat{x}$ | likelihood $p_\theta(x\mid z)$ |
| latent regularization | optional | KL to prior |
| sampling | not guaranteed meaningful | sample $z\sim p(z)$, decode |
| objective | reconstruction loss | ELBO |
| probabilistic model | usually no | yes |

The VAE can reconstruct like an autoencoder, but its key goal is generative modeling through a structured latent distribution.

## Reconstruction vs Generation

Reconstruction path:

$$
x
\rightarrow
q_\phi(z\mid x)
\rightarrow
p_\theta(x\mid z)
\rightarrow
\hat{x}.
$$

Generation path:

$$
z\sim p(z)
\rightarrow
p_\theta(x\mid z)
\rightarrow
\hat{x}.
$$

These are not equivalent.

Good reconstructions mean:

$$
q_\phi(z\mid x)
\text{ contains enough information to rebuild } x.
$$

Good prior samples mean:

$$
z\sim p(z)
\text{ lands in latent regions the decoder maps to valid data}.
$$

Many VAE failures come from confusing these two claims.

## Posterior Collapse

Posterior collapse occurs when:

$$
q_\phi(z\mid x)
\approx
p(z).
$$

Then $z$ carries little information about $x$:

$$
I_q(x;z)
\approx
0.
$$

This can happen when the decoder is powerful enough to model $x$ without using $z$, especially with autoregressive decoders.

Symptoms:

| Symptom | Meaning |
| --- | --- |
| KL near zero | latent ignored |
| reconstructions weakly depend on $z$ | decoder dominates |
| latent interpolation uninformative | code lacks semantic structure |
| prior samples generic | posterior and decoder fail to use latent information |

Posterior collapse is a central reading risk for any VAE-style paper.

## Amortization Gap

VAE uses a shared encoder:

$$
q_\phi(z\mid x)
$$

for all datapoints. This is efficient, but it can be less optimal than per-datapoint variational optimization.

The gap between the best possible variational posterior in a family and the encoder's output is the amortization gap.

Reading:

$$
\text{fast inference}
\quad
\text{trades against}
\quad
\text{posterior flexibility}.
$$

This matters when using VAEs as representation learners.

## Relation to GANs

| Axis | VAE | GAN |
| --- | --- | --- |
| objective | variational lower bound | adversarial game |
| encoder | explicit approximate posterior | usually absent in original GAN |
| likelihood | lower-bound oriented | no explicit likelihood |
| samples | often blurrier in early image VAEs | often sharper |
| latent space | structured by encoder and prior | structured by generator training |
| training | usually stable | can be unstable |

VAEs made latent-variable generative modeling trainable and interpretable. GANs later improved perceptual sharpness but changed the objective.

## Relation to DDPM

DDPM is also a latent-variable model, but the latent path is a long noising trajectory:

$$
x_1,\ldots,x_T.
$$

VAE:

$$
z
\rightarrow
x.
$$

DDPM:

$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0.
$$

Both use variational lower-bound reasoning, but DDPM's latent variables are tied to a fixed diffusion process.

## Relation to Normalizing Flows

Normalizing flows learn an invertible mapping:

$$
x = f_\theta(z),
\qquad
z=f_\theta^{-1}(x).
$$

This gives exact likelihood through change of variables:

$$
\log p_\theta(x)
=
\log p(z)
+
\log
\left|
\det
\frac{\partial f_\theta^{-1}}{\partial x}
\right|.
$$

VAE does not require invertibility. It uses approximate inference and a lower bound instead.

Tradeoff:

| Model | Strength | Cost |
| --- | --- | --- |
| VAE | flexible encoder/decoder, amortized inference | approximate likelihood bound |
| Flow | exact likelihood, invertible inference | architectural constraints from invertibility |

## Relation to Molecular and Protein Generation

VAEs became common in early molecule and protein generators because latent spaces support:

- interpolation;
- optimization in latent space;
- conditional decoding;
- property-guided search;
- representation learning.

But domain generation needs stricter checks:

| Object | Extra Validity Issue |
| --- | --- |
| SMILES | syntax and canonicalization |
| molecular graph | valence, charge, disconnected fragments |
| 3D conformer | bond length, chirality, energy |
| protein sequence | motifs, family leakage, function |
| protein structure | equivariance, geometry, residue mapping |

For computational biology, prior-sample validity matters more than reconstruction alone.

## Evidence Reading

The paper demonstrates efficient stochastic variational inference and shows that neural encoder-decoder latent-variable models can be trained with gradient methods.

Evidence should be read as:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| optimized variational bound | scalable learning for latent-variable models | best perceptual sample quality |
| learned latent representations | encoder can approximate posterior | semantic disentanglement |
| generated samples | model can sample from prior | domain validity in structured objects |
| reconstruction behavior | decoder and latent can represent data | prior samples are useful |

The main contribution is the training and inference recipe, not only a sample-quality benchmark.

## Implementation Notes

Core implementation checklist:

| Component | Check |
| --- | --- |
| encoder | outputs $\mu_\phi(x)$ and $\log\sigma_\phi^2(x)$ |
| reparameterization | samples $\epsilon$ independent of parameters |
| decoder | outputs likelihood parameters, not just pixels/tokens |
| KL | computed with correct prior and sign |
| reconstruction | matches data likelihood |
| sampling | evaluates $z\sim p(z)$, not only $z\sim q(z\mid x)$ |
| reporting | separates reconstruction, KL, ELBO, and sample metrics |

Common Gaussian implementation:

$$
\log\sigma^2_\phi(x)
=
\operatorname{encoder}_{\log v}(x).
$$

Then:

$$
\sigma_\phi(x)
=
\exp\left(\frac{1}{2}\log\sigma^2_\phi(x)\right).
$$

Numerical stability depends on how variance is parameterized.

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| posterior collapse | latent variable carries little information |
| over-regularization | KL dominates and reconstructions degrade |
| weak prior samples | random latent points decode poorly |
| decoder likelihood mismatch | objective does not match data type |
| blurry image samples | Gaussian decoder and pixel likelihood can average modes |
| invalid structured outputs | sequence/graph constraints not enforced |
| conflating reconstruction with generation | overstates generative quality |

## Common Misreadings

### "A VAE is just an autoencoder with Gaussian noise."

No. The VAE is a latent-variable generative model trained by a variational lower bound with an approximate posterior.

### "Good reconstruction means good generation."

No. Reconstruction uses $q_\phi(z\mid x)$, while generation uses $z\sim p(z)$.

### "The KL term is just regularization."

It acts like regularization, but probabilistically it matches the approximate posterior to the prior and controls the latent distribution used for sampling.

### "The latent space is automatically disentangled."

No. Disentanglement requires additional assumptions, objectives, or inductive biases and is not guaranteed by the VAE objective alone.

## Later-Paper Checklist

When reading VAE-style papers, ask:

- What is the prior $p(z)$?
- What is the encoder family $q_\phi(z\mid x)$?
- What is the decoder likelihood $p_\theta(x\mid z)$?
- Is the latent continuous, discrete, hierarchical, or structured?
- Is the reparameterization path valid?
- Is the KL term weighted, annealed, clipped, or free-bits adjusted?
- Is posterior collapse measured?
- Are prior samples evaluated separately from reconstructions?
- Are generated samples valid under domain constraints?
- Is downstream utility measured independently?

## Why It Matters

VAE gave deep learning a practical framework for probabilistic encoders, decoders, latent spaces, and approximate inference.

For this wiki, it connects:

$$
\text{probabilistic latent variables}
\rightarrow
\text{amortized inference}
\rightarrow
\text{neural generative models}
\rightarrow
\text{structured latent representations}.
$$

It is a foundation note for molecular generation, representation learning, latent diffusion, and any paper that claims a learned latent space.

## Limitations

The original VAE recipe has limits:

- approximate posterior may be weak;
- decoder may ignore latents;
- samples can be blurry or low-fidelity under simple likelihoods;
- prior samples can be invalid in structured domains;
- ELBO can improve while downstream sample utility does not;
- disentanglement is not guaranteed.

The defensible claim:

$$
\text{VAE}
\Rightarrow
\text{scalable neural variational inference for latent-variable generative models}.
$$

The overclaim to avoid:

$$
\text{VAE}
\Rightarrow
\text{automatically meaningful latent factors and high-quality samples}.
$$

## Connections

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/elbo|ELBO]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/architectures/autoencoder|Autoencoder]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[papers/architectures/ddpm|DDPM]]
- [[papers/generative-models/index|Generative Model Papers]]
- [[papers/architectures/index|Architecture papers]]
