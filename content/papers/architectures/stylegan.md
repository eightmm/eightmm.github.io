---
title: StyleGAN
aliases:
  - papers/stylegan
  - papers/style-based-generator-architecture
  - papers/generative-models/stylegan
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - vision
---

# StyleGAN

> The paper made style-based synthesis a canonical generator architecture for high-resolution image GANs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | A Style-Based Generator Architecture for Generative Adversarial Networks |
| Authors | Tero Karras, Samuli Laine, Timo Aila |
| Year | 2019 |
| Venue | CVPR 2019 |
| arXiv | [1812.04948](https://arxiv.org/abs/1812.04948) |
| CVF | [CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html) |
| Code | [NVlabs/stylegan](https://github.com/NVlabs/stylegan) |
| Status | verified |

## Question

[[papers/architectures/generative-adversarial-nets|GAN]] defines the adversarial game, but it does not specify a generator architecture that gives interpretable control over image attributes.

StyleGAN asks:

$$
\text{Can a GAN generator separate latent control into scale-specific styles?}
$$

The paper borrows from style transfer and changes the generator interface:

$$
z
\rightarrow
w
\rightarrow
\text{per-layer styles}
\rightarrow
\text{image}.
$$

## Main Claim

StyleGAN replaces the direct latent-to-image generator with:

1. a mapping network from latent $z$ to intermediate latent $w$;
2. learned constant input;
3. per-layer adaptive instance normalization;
4. stochastic noise injection at each resolution.

The durable architecture claim is:

$$
\text{mapping network}
+
\text{style modulation per layer}
+
\text{stochastic detail noise}
\Rightarrow
\text{scale-specific control and improved image synthesis}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | latent code $z\sim \mathcal{N}(0,I)$ |
| Intermediate | mapped latent $w=f(z)$ |
| Generator input | learned constant tensor rather than direct spatial projection from $z$ |
| Style injection | per-layer adaptive normalization parameters from $w$ |
| Noise injection | independent per-pixel noise for stochastic variation |
| Output | high-resolution image |
| Training | adversarial image generation |
| Main bias | disentangle coarse, middle, and fine image attributes by synthesis scale |

## Mapping Network

The mapping network transforms the sampled latent:

$$
w = f_\theta(z).
$$

Instead of feeding $z$ directly into the synthesis network, StyleGAN uses $w$ to produce per-layer style parameters.

The intuition:

$$
\mathcal{Z}
\xrightarrow{f_\theta}
\mathcal{W}
\xrightarrow{\text{styles}}
\text{synthesis network}.
$$

The intermediate space $\mathcal{W}$ is intended to be less entangled than the original sampling space.

## Style Modulation with AdaIN

For a feature map at layer $\ell$:

$$
x_\ell\in\mathbb{R}^{H_\ell\times W_\ell\times C_\ell},
$$

StyleGAN normalizes each channel and applies style-dependent scale and bias:

$$
\operatorname{AdaIN}(x_{\ell,c}, y_{\ell})
=
y_{\ell,c}^{(s)}
\frac{x_{\ell,c}-\mu(x_{\ell,c})}{\sigma(x_{\ell,c})}
+
y_{\ell,c}^{(b)}.
$$

where $y_\ell=A_\ell(w)$ is an affine transform of $w$ that produces style parameters for layer $\ell$.

The key point is that $w$ controls each synthesis scale separately:

$$
y_\ell = A_\ell(w).
$$

## Learned Constant Input

Classic generators often project a latent vector into an initial spatial feature map:

$$
z \rightarrow h_0.
$$

StyleGAN starts from a learned constant:

$$
h_0 = c,
\qquad
c\in\mathbb{R}^{4\times4\times C}.
$$

The latent code does not define the initial spatial tensor directly. It controls the synthesis through per-layer styles.

## Noise Injection

At layer $\ell$, StyleGAN injects spatial noise:

$$
x_\ell'
=
x_\ell
+
b_\ell n_\ell,
$$

where:

| Symbol | Meaning |
| --- | --- |
| $n_\ell$ | independent noise image broadcast across channels |
| $b_\ell$ | learned per-channel noise strength |

This separates stochastic details from persistent semantic attributes:

| Attribute Type | Typical Control Route |
| --- | --- |
| pose, identity, global shape | coarse styles |
| facial parts, texture layout | middle styles |
| freckles, hair strands, fine texture | noise and fine styles |

## Progressive Synthesis View

The generator builds images through increasing resolutions:

$$
4^2
\rightarrow
8^2
\rightarrow
16^2
\rightarrow
\cdots
\rightarrow
1024^2.
$$

The architecture uses the resolution hierarchy as a control hierarchy. Early layers affect coarse structure; later layers affect fine details.

This is why StyleGAN is important beyond face generation:

$$
\text{latent control}
\quad
\text{becomes}
\quad
\text{layer-wise synthesis control}.
$$

## Relation to GAN

GAN gives the adversarial training interface:

$$
\min_G \max_D
\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))].
$$

StyleGAN changes $G$:

$$
G(z)
=
S(c;\{A_\ell(f(z)), n_\ell\}_{\ell=1}^{L}),
$$

where $S$ is the synthesis network.

The discriminator and adversarial loss are not the main novelty. The generator architecture is.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| image quality metrics | style-based generator improves sample quality over strong GAN baselines |
| interpolation metrics | intermediate latent space improves interpolation behavior |
| style mixing experiments | different layers control different visual scales |
| noise analysis | stochastic variation can be separated from semantic style |

## Limits

- The paper is centered on high-resolution image generation, especially faces.
- Disentanglement is improved but not guaranteed in a causal or semantic sense.
- The method depends on adversarial training stability and dataset quality.
- Later StyleGAN2 changes normalization and training details to reduce artifacts, so this note should be read as the original architecture milestone.

## Concepts

- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/latent-variable-model|Latent-variable model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/architectures/normalization|Normalization]]

## Related

- [[papers/architectures/generative-adversarial-nets|Generative Adversarial Nets]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
