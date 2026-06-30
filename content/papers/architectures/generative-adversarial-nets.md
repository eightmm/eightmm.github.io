---
title: Generative Adversarial Nets
aliases:
  - papers/gan
  - papers/generative-adversarial-nets
  - papers/generative-models/gan
tags:
  - papers
  - architectures
  - generative-models
  - gan
  - adversarial-training
---

# Generative Adversarial Nets

> The paper introduced adversarial generative modeling: train a generator to produce samples that a discriminator cannot distinguish from real data.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Generative Adversarial Nets |
| Authors | Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio |
| Year | 2014 |
| Venue | NeurIPS 2014 |
| arXiv | [1406.2661](https://arxiv.org/abs/1406.2661) |
| NeurIPS | [paper page](https://papers.nips.cc/paper/5423-generative-adversarial-nets) |
| Status | full note started |

## One-Line Takeaway

GAN replaces explicit likelihood training with a two-player game: a generator learns to map noise to samples, while a discriminator learns to tell generated samples from real data.

## Question

Generative modeling asks for a model distribution:

$$
p_\theta(x)
\approx
p_{\text{data}}(x).
$$

Many earlier models required one of the following:

- explicit likelihood;
- tractable latent-variable bounds;
- Markov chain sampling;
- hand-designed reconstruction losses;
- approximate inference networks.

GAN asks:

> Can a generator learn a data distribution using only feedback from a learned discriminator?

The basic sampling path:

$$
z\sim p(z),
\qquad
\hat{x}=G_\theta(z).
$$

The discriminator:

$$
D_\psi(x)\in(0,1)
$$

estimates whether $x$ came from real data rather than the generator.

## Main Claim

The narrowed claim:

$$
\text{generator}
+
\text{discriminator}
+
\text{minimax adversarial objective}
\Rightarrow
\text{implicit generative modeling without explicit likelihood}.
$$

In the idealized setting with enough capacity and optimal training, the generator distribution can match the data distribution:

$$
p_g = p_{\text{data}},
\qquad
D(x)=\frac{1}{2}.
$$

The practical claim is weaker:

$$
\text{adversarial training}
\Rightarrow
\text{sharp sample generation but difficult optimization}.
$$

## Architecture Contract

| Component | Role |
| --- | --- |
| latent prior $p(z)$ | simple noise source |
| generator $G_\theta$ | maps latent noise to generated samples |
| discriminator $D_\psi$ | predicts whether a sample is real or generated |
| adversarial objective | couples the two networks in a minimax game |
| sampling interface | draw $z$, return $G_\theta(z)$ |
| evaluation | sample quality, diversity, novelty, downstream utility |

The key difference from VAE:

$$
\text{VAE: likelihood lower bound}
\qquad
\text{GAN: discriminator-defined training signal}.
$$

The key difference from DDPM:

$$
\text{DDPM: iterative denoising}
\qquad
\text{GAN: one-shot generation}.
$$

## Generator

The generator transforms latent noise into samples:

$$
G_\theta:\mathcal{Z}\rightarrow\mathcal{X}.
$$

Sampling:

$$
z\sim p(z),
\qquad
\hat{x}=G_\theta(z).
$$

This defines an implicit model distribution:

$$
\hat{x}\sim p_g.
$$

Usually:

$$
\log p_g(x)
$$

is not tractable. Samples are easy; exact density evaluation is not.

This is the core architectural tradeoff:

$$
\text{fast sample generation}
\quad
\text{without}
\quad
\text{explicit likelihood}.
$$

## Discriminator

The discriminator maps a sample to a probability:

$$
D_\psi(x)
\approx
P(y=\text{real}\mid x).
$$

It is trained on:

- real samples $x\sim p_{\text{data}}$;
- generated samples $\hat{x}=G_\theta(z)$.

The discriminator objective is to separate the two:

$$
D_\psi(x)\rightarrow 1
\quad
\text{for real } x,
$$

$$
D_\psi(G_\theta(z))\rightarrow 0
\quad
\text{for generated samples}.
$$

The discriminator is not just an evaluator after training. It defines the learning signal for the generator.

## Minimax Objective

The original GAN objective:

$$
\min_G \max_D
V(D,G)
$$

where:

$$
V(D,G)
=
\mathbb{E}_{x\sim p_{\text{data}}}
[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}
[\log(1-D(G(z)))].
$$

The discriminator maximizes:

$$
\mathbb{E}_{x\sim p_{\text{data}}}
[\log D(x)]
+
\mathbb{E}_{z\sim p(z)}
[\log(1-D(G(z)))].
$$

The generator minimizes:

$$
\mathbb{E}_{z\sim p(z)}
[\log(1-D(G(z)))].
$$

The generator wants:

$$
D(G(z))\rightarrow 1.
$$

The discriminator wants:

$$
D(G(z))\rightarrow 0.
$$

This opposing objective is why GAN training is a game, not a single supervised loss.

## Optimal Discriminator

For a fixed generator distribution $p_g$, the optimal discriminator is:

$$
D^*(x)
=
\frac{
p_{\text{data}}(x)
}{
p_{\text{data}}(x)+p_g(x)
}.
$$

If:

$$
p_g(x)=p_{\text{data}}(x),
$$

then:

$$
D^*(x)=\frac{1}{2}.
$$

Interpretation:

$$
\text{perfect generator}
\Rightarrow
\text{discriminator cannot distinguish real and fake}.
$$

This gives GANs a clean theoretical target. The hard part is optimization with finite networks, finite data, and alternating gradient updates.

## Jensen-Shannon Reading

With the optimal discriminator, the original GAN objective is related to the Jensen-Shannon divergence:

$$
C(G)
=
-\log 4
+
2\,D_{\mathrm{JS}}
\left(
p_{\text{data}}
\Vert
p_g
\right).
$$

Thus minimizing the idealized objective pushes:

$$
p_g
\rightarrow
p_{\text{data}}.
$$

But this elegant result assumes the discriminator is optimal and the game can be solved. Practical training often violates both assumptions.

## Non-Saturating Generator Loss

The minimax generator loss can saturate when the discriminator is too good. A common practical alternative is the non-saturating loss:

$$
\mathcal{L}_G
=
-
\mathbb{E}_{z\sim p(z)}
[\log D(G(z))].
$$

The discriminator loss can be written as:

$$
\mathcal{L}_D
=
-
\mathbb{E}_{x\sim p_{\text{data}}}
[\log D(x)]
-
\mathbb{E}_{z\sim p(z)}
[\log(1-D(G(z)))].
$$

This changes gradient behavior while preserving the intuition:

$$
\text{make fake samples look real to }D.
$$

When reading GAN papers, always check the exact loss. "GAN" is a family of objectives, not just one equation.

## Training Loop

A simplified training loop:

1. sample real minibatch $x\sim p_{\text{data}}$;
2. sample latent noise $z\sim p(z)$;
3. generate $\hat{x}=G_\theta(z)$;
4. update $D_\psi$ to classify $x$ as real and $\hat{x}$ as fake;
5. sample new $z$;
6. update $G_\theta$ to fool $D_\psi$.

The update ratio matters:

$$
n_D:n_G.
$$

If the discriminator is too weak, the generator receives poor guidance. If the discriminator is too strong, generator gradients may become unhelpful.

## Implicit Distribution

GANs produce samples without giving a tractable density:

$$
z\sim p(z),
\qquad
x=G_\theta(z),
\qquad
x\sim p_g.
$$

But usually:

$$
p_g(x)
\quad
\text{is not explicitly computable}.
$$

This affects evaluation. For VAEs or flows, one can discuss likelihood bounds or exact likelihood. For GANs, one relies heavily on sample-based checks.

## Fidelity and Coverage

GANs can produce sharp samples while missing modes.

Fidelity:

$$
\text{do samples look real?}
$$

Coverage:

$$
\text{does }p_g\text{ cover all important modes of }p_{\text{data}}?
$$

Mode collapse:

$$
z_1,z_2,\ldots,z_n
\rightarrow
\text{same or narrow set of outputs}.
$$

This is a core GAN failure mode. A small curated sample grid can look excellent while distribution coverage is poor.

## Relation to VAEs

| Axis | VAE | GAN |
| --- | --- | --- |
| training signal | ELBO | adversarial discriminator |
| encoder | explicit $q_\phi(z\mid x)$ | absent in original GAN |
| decoder/generator | likelihood model $p_\theta(x\mid z)$ | implicit generator $G_\theta(z)$ |
| likelihood | lower bound available | usually unavailable |
| samples | often smoother in early image VAEs | often sharper |
| training stability | usually easier | often unstable |
| main failure | posterior collapse, weak prior samples | mode collapse, non-convergence |

VAE asks:

$$
\text{Can we optimize a probabilistic latent-variable bound?}
$$

GAN asks:

$$
\text{Can a learned critic provide the training signal?}
$$

## Relation to DDPM

| Axis | GAN | DDPM |
| --- | --- | --- |
| sampling | one forward generator pass | many denoising steps |
| objective | adversarial game | denoising/variational objective |
| density | implicit | variational/score interpretation |
| training stability | difficult | usually more stable |
| sample speed | fast | slower unless accelerated |
| sample quality | sharp with collapse risk | high quality with sampling cost |

GANs made high-fidelity generation visible. Diffusion later changed the tradeoff by favoring stable denoising training and strong mode coverage at the cost of sampling steps.

## Relation to Normalizing Flows

Flows define an invertible map:

$$
x=f_\theta(z),
\qquad
z=f_\theta^{-1}(x).
$$

Likelihood is available through:

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

GAN generators are not required to be invertible and usually do not give exact likelihood.

Tradeoff:

$$
\text{architectural freedom}
\quad
\text{vs.}
\quad
\text{density evaluation}.
$$

## Conditional GANs

A conditional GAN adds context $c$:

$$
\hat{x}
=
G_\theta(z,c),
$$

$$
D_\psi(x,c)
\rightarrow
P(x\text{ real}\mid c).
$$

Condition types:

- class label;
- text prompt;
- image;
- segmentation map;
- molecule scaffold;
- protein pocket;
- property target.

Conditional evaluation must ask:

$$
\text{is the sample both realistic and condition-satisfying?}
$$

A realistic sample that ignores $c$ is a conditional failure.

## Evaluation Reading

GAN evaluation is difficult because exact likelihood is unavailable.

| Evaluation Axis | Question | Risk |
| --- | --- | --- |
| visual/sample quality | do samples look realistic? | cherry-picked samples |
| diversity | do samples cover modes? | mode collapse hidden |
| novelty | are samples copied from train set? | memorization |
| conditional accuracy | does $x$ satisfy $c$? | condition ignored |
| downstream utility | does generated data help a task? | evaluator bias |
| sample efficiency | how many attempts per useful sample? | hidden rejection |

For structured domains, count invalid samples:

$$
\text{validity}
=
\frac{
\#\text{valid generated samples}
}{
\#\text{attempted generated samples}
}.
$$

Do not report metrics only after filtering invalid outputs unless the filter is part of the method and counted.

## Molecular and Scientific Modeling Reading

GANs inspired many molecule/protein generators, but scientific domains add constraints:

| Object | GAN Risk |
| --- | --- |
| molecule graph | valence, charge, stereochemistry errors |
| SMILES | invalid syntax or duplicate strings |
| conformer | bad bond geometry, chirality, high energy |
| protein sequence | family leakage or motif copying |
| 3D structure | equivariance and physical plausibility |
| ligand pose | pose realism without affinity evidence |

Adversarial realism is not the same as scientific validity:

$$
\text{fools discriminator}
\neq
\text{chemically correct}
\neq
\text{biologically useful}.
$$

This distinction is essential for public paper notes.

## Stabilization Patterns

Later GAN papers introduced many stabilization techniques:

| Pattern | Purpose | Reading Risk |
| --- | --- | --- |
| feature matching | reduce instability | changes generator objective |
| minibatch discrimination | expose lack of diversity | can be dataset-specific |
| gradient penalty | regularize discriminator smoothness | adds compute and objective assumptions |
| spectral normalization | constrain discriminator Lipschitz behavior | affects capacity |
| Wasserstein objective | improve distance behavior | requires critic constraints |
| two-time-scale updates | balance game dynamics | update ratio becomes a method parameter |

The original paper is the conceptual anchor, but later GAN practice depends heavily on these refinements.

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| mode collapse | generator covers narrow output modes |
| non-convergence | game dynamics oscillate or diverge |
| discriminator overpowering | generator gradient becomes weak or misleading |
| memorization | samples copy training examples |
| metric gaming | generator exploits evaluator weaknesses |
| hidden filtering | low-quality samples are discarded before reporting |
| conditional ignoring | output looks real but ignores condition |
| invalid scientific outputs | discriminator realism misses domain constraints |

## Common Misreadings

### "GANs learn likelihood-free generation, so evaluation is easy."

No. Likelihood-free generation makes evaluation harder. Samples are easy to draw, but density and coverage are difficult to measure.

### "If samples look sharp, the model is good."

No. Sharpness measures fidelity, not coverage, novelty, condition satisfaction, or downstream utility.

### "The discriminator is a perfect learned metric."

No. The discriminator is trained in a finite game and can miss domain-specific invalidity or overfit to dataset artifacts.

### "GAN is one architecture."

No. GAN is a training framework. The generator and discriminator architectures, loss variant, conditioning, and stabilization method define the actual model.

## Later-Paper Checklist

When reading later GAN papers, check:

- What generator architecture is used?
- What discriminator or critic architecture is used?
- Is the loss minimax, non-saturating, hinge, Wasserstein, or another variant?
- How many discriminator updates per generator update?
- Is conditioning used, and how is condition satisfaction measured?
- Are sample grids random or curated?
- Are invalid samples counted?
- Is mode coverage measured?
- Is nearest-neighbor memorization checked?
- Are metrics computed with the same sample budget as baselines?
- For scientific outputs, are domain validity checks independent of the discriminator?

## Why It Matters

GAN is a foundation paper because it introduced a new way to train generative models:

$$
\text{learn a generator}
\quad
\text{by competing with}
\quad
\text{a learned discriminator}.
$$

For this wiki, GAN connects:

$$
\text{implicit generative models}
\rightarrow
\text{adversarial objectives}
\rightarrow
\text{fidelity/coverage evaluation}
\rightarrow
\text{conditional generation risks}.
$$

It should sit beside VAE and DDPM because the three papers define three different generative modeling contracts:

| Paper | Contract |
| --- | --- |
| [[papers/architectures/auto-encoding-variational-bayes|VAE]] | latent-variable model with amortized inference and ELBO |
| GAN | implicit generator trained by discriminator feedback |
| [[papers/architectures/ddpm|DDPM]] | iterative denoising model trained by noising-reversal objective |

## Limitations

The original GAN framework has important limitations:

- difficult game optimization;
- mode collapse;
- no tractable likelihood;
- unstable sensitivity to architecture and hyperparameters;
- weak guarantees with finite networks;
- evaluation difficulty;
- domain validity not guaranteed by adversarial realism.

The defensible claim:

$$
\text{GAN}
\Rightarrow
\text{powerful adversarial framework for implicit generative modeling}.
$$

The overclaim to avoid:

$$
\text{GAN}
\Rightarrow
\text{easy, stable, complete distribution learning}.
$$

## Connections

- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/ddpm|DDPM]]
- [[papers/generative-models/index|Generative Model Papers]]
- [[papers/architectures/index|Architecture papers]]
