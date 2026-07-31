---
title: Neural Discrete Representation Learning
aliases:
  - papers/vq-vae
  - papers/neural-discrete-representation-learning
  - papers/vector-quantised-variational-autoencoder
tags:
  - papers
  - architectures
  - generative-models
  - vae
  - discrete-latent
  - vector-quantization
---

# Neural Discrete Representation Learning

> VQ-VAE replaces continuous Gaussian latents with a learned discrete codebook, making latent generative modeling look like compression plus an autoregressive prior over codes.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Discrete Representation Learning |
| Authors | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1711.00937](https://arxiv.org/abs/1711.00937) |
| Proceedings | [NeurIPS 2017](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning) |
| Status | full note |

## One-Line Takeaway

VQ-VAE learns a discrete latent codebook and trains an encoder to choose nearest code vectors, allowing a decoder to reconstruct from discrete symbols and a separate autoregressive prior to generate in latent space.

## Question

Standard [[papers/architectures/auto-encoding-variational-bayes|VAE]] uses continuous latent variables:

$$
z \sim \mathcal{N}(0,I),
\qquad
x \sim p_\theta(x\mid z).
$$

With a powerful decoder, the model can ignore $z$:

$$
p_\theta(x\mid z) \approx p_\theta(x).
$$

This is posterior collapse. VQ-VAE asks:

> Can a generative model learn useful discrete latent representations while avoiding the tendency of a powerful decoder to ignore the latent code?

## Architecture Contract

| Component | Role |
| --- | --- |
| encoder | maps input to continuous latent vectors |
| codebook | stores learned discrete embedding vectors |
| vector quantization | replaces each encoder output with nearest codebook entry |
| decoder | reconstructs input from quantized code vectors |
| straight-through estimator | routes gradients from decoder to encoder |
| commitment loss | keeps encoder outputs close to selected codes |
| autoregressive prior | models the sequence or grid of discrete code indices |

The model has two paths:

$$
x
\rightarrow
z_e(x)
\rightarrow
z_q(x)
\rightarrow
\hat{x}
$$

for reconstruction, and:

$$
k_{1:N}\sim p_\psi(k_{1:N}),
\qquad
z_q = e_{k_{1:N}},
\qquad
x\sim p_\theta(x\mid z_q)
$$

for generation.

## Codebook Quantization

Let the encoder output be:

$$
z_e(x) \in \mathbb{R}^{D}.
$$

Let the codebook be:

$$
E=\{e_1,\ldots,e_K\},
\qquad
e_k\in\mathbb{R}^{D}.
$$

The discrete assignment is nearest-neighbor lookup:

$$
k^\*
=
\arg\min_{k}
\lVert z_e(x)-e_k\rVert_2.
$$

The quantized latent is:

$$
z_q(x) = e_{k^\*}.
$$

For an image, this usually happens over a spatial latent grid:

$$
z_e(x) \in \mathbb{R}^{H'\times W'\times D},
\qquad
k_{u,v}\in\{1,\ldots,K\}.
$$

So the latent representation becomes a grid of discrete symbols.

## Objective

The VQ-VAE loss has three terms:

$$
\mathcal{L}
=
-\log p_\theta(x\mid z_q(x))
+
\lVert
\operatorname{sg}[z_e(x)] - e
\rVert_2^2
+
\beta
\lVert
z_e(x)-\operatorname{sg}[e]
\rVert_2^2.
$$

where:

- $\operatorname{sg}[\cdot]$ is stop-gradient;
- the first term trains the decoder;
- the second term moves codebook embeddings toward encoder outputs;
- the third term makes the encoder commit to a code.

The straight-through estimator copies the decoder gradient through the quantization operation:

$$
\frac{\partial z_q}{\partial z_e}
\approx
I.
$$

This is not mathematically exact differentiation through nearest-neighbor lookup. It is a practical estimator that makes the architecture trainable.

## Learned Prior

After learning discrete latents, train a prior over code indices:

$$
p_\psi(k_{1:N})
=
\prod_{i=1}^{N}
p_\psi(k_i\mid k_{<i}).
$$

The prior can be a [[papers/architectures/pixel-recurrent-neural-networks|PixelCNN]] over the latent grid.

This changes the hard problem:

$$
\text{generate high-dimensional pixels directly}
$$

into:

$$
\text{generate a lower-resolution grid of discrete codes}
\rightarrow
\text{decode to pixels}.
$$

That pattern later becomes central in latent generative modeling.

## VQ-VAE vs VAE

| Axis | VAE | VQ-VAE |
| --- | --- | --- |
| latent type | continuous Gaussian | discrete codebook index |
| posterior | parametric distribution $q_\phi(z\mid x)$ | nearest-neighbor code assignment |
| regularization | KL to prior | codebook and commitment losses |
| prior | often fixed Gaussian during base training | learned prior over discrete codes |
| collapse risk | high with powerful decoders | reduced by discrete bottleneck |

VQ-VAE is still a latent-variable generative architecture, but the bottleneck is closer to learned vector quantization than a Gaussian posterior.

## Why It Matters

VQ-VAE is a bridge between autoencoders, compression, tokenization, and generative modeling.

| Contribution | Later Use |
| --- | --- |
| learned discrete visual/audio codes | tokenizer-like representations for non-text data |
| codebook bottleneck | controllable compression and discrete latent structure |
| autoregressive prior over codes | faster generation than raw-pixel autoregression |
| separation of tokenizer and prior | precursor to many latent generative pipelines |

For modern architecture reading, VQ-VAE is important because it makes a visual or audio object look like a sequence/grid of tokens.

## What To Watch

- Codebook collapse can occur when only a few codes are used.
- Reconstruction quality and generative sample quality are separate claims.
- The learned prior matters; the autoencoder alone is not a full generative model.
- Discrete codes can hide artifacts if the decoder is too strong or the bottleneck is poorly sized.
- Later latent diffusion models use continuous latent autoencoders, but the same “compress then generate in latent space” design logic is shared.

## Encoder and decoder contract

Let an input $x$ be mapped by an encoder to a continuous latent grid:

$$
z_e(x)
=
f_\phi(x)
\in
\mathbb{R}^{M\times D},
$$

where $M$ is the number of latent locations and $D$ is the code dimension. Each location is quantized independently:

$$
k_i
=
\arg\min_{k\in\{1,\ldots,K\}}
\left\|
z_{e,i}(x)-e_k
\right\|_2^2,
$$

and the decoder receives:

$$
z_{q,i}(x)
=
e_{k_i}.
$$

The reconstruction path is:

$$
x
\xrightarrow{f_\phi}
z_e
\xrightarrow{Q_E}
z_q
\xrightarrow{g_\theta}
\hat x.
$$

The code indices $k_{1:M}$ are the discrete representation. The vectors $e_{k_i}$ are decoder-facing continuous values looked up from that representation.

This distinction matters in implementations:

| Object | Type | Stored or trained as |
| --- | --- | --- |
| code index $k_i$ | integer | discrete token |
| code vector $e_{k_i}$ | real vector | codebook lookup |
| encoder output $z_{e,i}$ | real vector | continuous pre-quantization feature |
| decoder input $z_{q,i}$ | real vector | selected code vector |

Calling $z_q$ a continuous latent without recording the underlying indices hides the part of the architecture that later enables autoregressive token modeling.

## Codebook geometry

The codebook is:

$$
E
\in
\mathbb{R}^{K\times D}.
$$

Each row is a prototype in the encoder feature space. Quantization partitions that space into Voronoi cells:

$$
\mathcal{V}_k
=
\left\{
z:
\|z-e_k\|_2
\le
\|z-e_j\|_2
\;\forall j
\right\}.
$$

The encoder output is represented by the prototype of the cell it falls into. The number of codes $K$ controls the dictionary size, while $D$ controls the dimension of each code vector.

The nominal information capacity of a length-$M$ code sequence is:

$$
M\log_2 K
\quad\text{bits}
$$

before accounting for code usage imbalance and the entropy of the learned prior. A larger codebook does not guarantee higher effective capacity if only a small fraction of its entries are used.

The empirical code usage distribution is:

$$
\hat p(k)
=
\frac{1}{M_{\mathrm{data}}}
\sum_{i=1}^{M_{\mathrm{data}}}
\mathbf{1}[k_i=k].
$$

Its perplexity is:

$$
\operatorname{Perplexity}(E)
=
\exp
\left(
-\sum_{k=1}^{K}
\hat p(k)\log\hat p(k)
\right).
$$

Perplexity near one indicates collapse to very few codes. Perplexity near $K$ indicates broad usage, but broad usage alone does not prove that the codes are semantically useful.

## The three loss terms in detail

For one encoder vector $z_e$ and selected code $e_{k^\*}$, the VQ-VAE objective is:

$$
\mathcal{L}_{\mathrm{VQ}}
=
\underbrace{
-\log p_\theta(x\mid z_q)
}_{\text{reconstruction}}
+
\underbrace{
\left\|
\operatorname{sg}[z_e]-e_{k^\*}
\right\|_2^2
}_{\text{codebook update}}
+
\beta
\underbrace{
\left\|
z_e-\operatorname{sg}[e_{k^\*}]
\right\|_2^2
}_{\text{commitment}}.
$$

The stop-gradient operator:

$$
\operatorname{sg}[u]
=
u
\quad\text{in the forward pass},
\qquad
\frac{\partial\operatorname{sg}[u]}{\partial u}=0
$$

assigns different optimization responsibility to the two terms:

| Term | Encoder gradient | Codebook gradient | Purpose |
| --- | --- | --- | --- |
| reconstruction | through straight-through path | indirect through decoder path | preserve input information |
| codebook | blocked by $\operatorname{sg}[z_e]$ | moves $e_k$ toward encoder outputs | place prototypes near assigned features |
| commitment | pulls $z_e$ toward $e_k$ | blocked by $\operatorname{sg}[e_k]$ | prevent encoder outputs from drifting |

The coefficient $\beta$ is not a harmless regularization constant. If it is too small, encoder outputs can move between code vectors and cause unstable assignments. If it is too large, the encoder may be forced into an overly rigid code geometry before reconstruction is adequate.

## Straight-through estimator

The quantizer:

$$
z_q
=
e_{\arg\min_k\|z_e-e_k\|_2}
$$

is piecewise constant with respect to $z_e$ almost everywhere, so its exact derivative is not useful for ordinary backpropagation. The implementation uses:

$$
z_q^{\mathrm{st}}
=
z_e
+
\operatorname{sg}(z_q-z_e).
$$

In the forward pass:

$$
z_q^{\mathrm{st}}=z_q,
$$

while in the backward pass:

$$
\frac{\partial z_q^{\mathrm{st}}}{\partial z_e}
=
I.
$$

This estimator sends the decoder's reconstruction gradient to the encoder as if the quantization were the identity. It does not make nearest-neighbor assignment differentiable in the exact mathematical sense.

That distinction should be stated in a reproduction note because alternative quantizers may use soft assignments, Gumbel relaxation, EMA codebook updates, or other gradient estimators.

## Codebook update variants

The original loss updates codebook vectors with gradient descent. An alternative is an exponential moving average update. For assigned encoder vectors with count $n_k$ and sum $m_k$:

$$
N_k^{(t)}
=
\gamma N_k^{(t-1)}
+
(1-\gamma)n_k^{(t)},
$$

$$
M_k^{(t)}
=
\gamma M_k^{(t-1)}
+
(1-\gamma)m_k^{(t)},
$$

then:

$$
e_k^{(t)}
=
\frac{M_k^{(t)}}{N_k^{(t)}+\epsilon}.
$$

EMA updates change the optimization dynamics. They are not interchangeable with the codebook loss without reporting the update rule, initialization, smoothing, and handling of unused codes.

## Why discretization can prevent posterior collapse

In a continuous VAE, a strong decoder can model $x$ without relying on $z$. The KL term can then encourage the posterior toward the prior while the decoder ignores the latent.

VQ-VAE changes the bottleneck:

$$
z_e(x)
\rightarrow
k(x)
\rightarrow
e_{k(x)}.
$$

The decoder receives a finite code selected by the encoder. The codebook and commitment objectives encourage the encoder to use the bottleneck consistently. This can make the discrete representation informative even when the decoder is expressive.

The claim should still be qualified. Discretization does not guarantee that the latent is used for every task, and a decoder can still exploit shortcuts in the code or input pipeline. The relevant evidence is reconstruction, code usage, downstream utility, and the quality of the learned prior together.

## The autoregressive prior

After the tokenizer is trained, the encoder-decoder can be frozen and a prior can be trained over indices:

$$
p_\psi(k_{1:M})
=
\prod_{i=1}^{M}
p_\psi(k_i\mid k_{<i}).
$$

For a two-dimensional latent grid, the ordering used to flatten the grid is part of the prior contract:

$$
(k_{1,1},k_{1,2},\ldots,k_{H',W'})
\rightarrow
(k_1,\ldots,k_M).
$$

The prior may be a PixelCNN, Transformer, or another sequence model. The full generative process is:

$$
k_{1:M}
\sim
p_\psi(k_{1:M}),
\qquad
z_q=e_{k_{1:M}},
\qquad
x\sim p_\theta(x\mid z_q).
$$

The computational advantage comes from reducing the sequence length:

$$
M
=
\frac{H'W'}{1}
\ll
HW
$$

for a spatially downsampled latent grid. The decoder must then restore the missing detail from the discrete code.

The tokenizer and prior are separate failure surfaces:

| Failure | Symptom |
| --- | --- |
| poor tokenizer | generated code decodes to low-quality or invalid samples |
| poor prior | code sequences are implausible even when the tokenizer reconstructs well |
| prior overfitting | training reconstruction is good but sampled outputs lack diversity |
| ordering mismatch | prior learns an artificial dependency pattern |

## Compression and generation are different claims

A VQ-VAE can be useful as a compressor without producing high-quality unconditional samples. Conversely, a powerful prior can generate plausible code sequences while the decoder produces blurry or invalid outputs.

Separate the claims:

$$
\text{representation quality}
\ne
\text{reconstruction quality}
\ne
\text{prior likelihood}
\ne
\text{sample quality}.
$$

For a complete evaluation, report:

- reconstruction loss or perceptual reconstruction metrics;
- codebook usage and perplexity;
- latent compression ratio;
- prior negative log-likelihood or bits per code;
- unconditional sample quality;
- conditional generation quality if a condition is used;
- diversity and mode coverage;
- decoder compute and sampling cost.

## Relation to PixelCNN

Raw-pixel autoregressive models factorize:

$$
p(x)
=
\prod_{i=1}^{HW}
p(x_i\mid x_{<i}).
$$

VQ-VAE changes the factorization:

$$
p(x)
=
\sum_{k}
p_\theta(x\mid k)p_\psi(k).
$$

The decoder models $p_\theta(x\mid k)$ and the prior models $p_\psi(k)$. The prior sees a shorter sequence, while the decoder handles local detail conditioned on the code.

| Design | Sequence modeled by prior | Main cost |
| --- | --- | --- |
| PixelCNN | pixels | long sequential sampling |
| VQ-VAE + PixelCNN | discrete latent codes | code prior plus decoder |
| VQGAN + Transformer | learned image codes | tokenizer, Transformer prior, decoder |
| latent diffusion | continuous latent states | iterative denoising in latent space |

The common architecture pattern is “learn a representation interface first, then select a generative process over that interface.”

## Relation to later latent models

VQ-VAE's discrete codebook is one route to a learned tokenizer. Later systems may choose a continuous autoencoder, a discrete VQGAN tokenizer, or a multimodal token interface. The underlying design questions remain:

1. What object does one latent token represent?
2. How much spatial or temporal detail is discarded?
3. Is the latent interface invertible, approximately reconstructive, or task-specific?
4. Which model learns the prior over latent states?
5. Which metrics evaluate the representation separately from the generator?

These questions connect to [[papers/architectures/taming-transformers|Taming Transformers]], [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]], and [[papers/architectures/scalable-diffusion-models-with-transformers|DiT]].

## Ablation questions

- How does codebook size $K$ affect reconstruction, perplexity, and prior modeling?
- How does latent grid resolution affect compression and sample quality?
- Does EMA codebook updating differ from gradient codebook learning under matched compute?
- How sensitive is the model to commitment coefficient $\beta$?
- How many codes are active on train and held-out data?
- Does a larger decoder hide a poor codebook?
- Does the autoregressive prior benefit from a different rasterization or ordering?
- What happens when the tokenizer is frozen before prior training versus jointly adapted?
- Does code-level likelihood correlate with sample quality?
- How does discrete latent modeling compare with a continuous autoencoder at equal bitrate?

## Failure modes and diagnostics

### Codebook collapse

Only a small subset of codes is selected:

$$
\left|\{k:\hat p(k)>0\}\right|
\ll K.
$$

Inspect usage histograms, perplexity, dead-code count, and code assignments by data subset. Increasing $K$ alone does not solve collapse.

### Dead codes

A code may receive no assignments and therefore no useful update. Initialization, EMA smoothing, restart rules, and batch diversity can affect this.

### Encoder commitment instability

If assignments change rapidly, reconstruction may oscillate and the prior sees a non-stationary token vocabulary. Track assignment entropy and codebook movement across training.

### Decoder shortcut

A decoder with excessive capacity may reconstruct well from weak codes. Compare reconstruction after code dropout or code shuffling, and evaluate downstream tasks using the codes without the decoder.

### Prior-token mismatch

The prior must use the same code indices and spatial ordering as the tokenizer. A checkpoint with a changed codebook cannot be paired with an old prior without an explicit remapping or retraining procedure.

## Reproduction checklist

- [ ] Record encoder downsampling ratio, latent grid shape, code dimension, and codebook size.
- [ ] Specify nearest-neighbor distance, tie handling, and code initialization.
- [ ] State the exact stop-gradient and straight-through implementation.
- [ ] Report commitment coefficient and codebook update rule.
- [ ] Track active code count, perplexity, dead codes, and assignment entropy.
- [ ] Separate tokenizer training from prior training in the experiment description.
- [ ] Record latent flattening order and prior context length.
- [ ] Evaluate reconstruction, compression, code likelihood, and samples separately.
- [ ] Test held-out data to detect codebook collapse hidden by training reconstruction.
- [ ] Report decoder and prior compute, not only parameter count.

## Takeaway

VQ-VAE turns a high-dimensional object into a learned discrete interface:

$$
\boxed{
x
\rightarrow
\text{encoder}
\rightarrow
\text{code index grid}
\rightarrow
\text{decoder}
}
$$

The architecture's durable lesson is that representation learning and generation can be decoupled. A tokenizer defines the symbols; a prior defines how symbols are composed; a decoder turns them back into the target space. Each layer has its own objective and failure modes, so a good paper note must not collapse them into one sample-quality number.

## Related

- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[papers/architectures/auto-encoding-variational-bayes|Auto-Encoding Variational Bayes]]
- [[papers/architectures/pixel-recurrent-neural-networks|PixelRNN / PixelCNN]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
