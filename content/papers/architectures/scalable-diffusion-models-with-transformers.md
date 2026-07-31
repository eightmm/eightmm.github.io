---
title: Scalable Diffusion Models with Transformers
aliases:
  - papers/dit
  - papers/diffusion-transformer
  - papers/scalable-diffusion-models-with-transformers
  - papers/generative-models/dit
tags:
  - papers
  - architectures
  - generative-models
  - diffusion
  - transformer
---

# Scalable Diffusion Models with Transformers

> The paper replaces the usual convolutional U-Net denoiser in latent diffusion with a Transformer over latent image patches.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Scalable Diffusion Models with Transformers |
| Authors | William Peebles, Saining Xie |
| Year | 2022 preprint; 2023 conference |
| Venue | ICCV 2023 |
| arXiv | [2212.09748](https://arxiv.org/abs/2212.09748) |
| Project | [DiT project page](https://www.wpeebles.com/DiT.html) |
| Code | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) |
| Status | full note |

## Question

[[papers/architectures/ddpm|DDPM]] and [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]] often use convolutional U-Net denoisers. DiT asks:

$$
\text{Can a Transformer be the main denoising backbone for image diffusion?}
$$

The paper's answer:

$$
\text{latent image}
\rightarrow
\text{patch tokens}
\rightarrow
\text{Transformer denoiser}
\rightarrow
\text{noise or velocity prediction}.
$$

## Main Claim

Diffusion Transformers replace the U-Net backbone with a Transformer operating on latent patches and show predictable scaling with model compute.

The durable architecture claim is:

$$
\text{latent diffusion}
+
\text{ViT-style patch tokens}
+
\text{conditional Transformer blocks}
\Rightarrow
\text{scalable diffusion backbone}.
$$

This is an architecture paper because it changes the denoising network family, not the basic diffusion objective.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | noisy latent image $z_t\in\mathbb{R}^{h\times w\times c}$ |
| Tokenization | split latent into patches |
| Backbone | Transformer blocks |
| Conditioning | timestep and class conditioning |
| Output | predicted noise, velocity, or denoising target over latent patches |
| Decoder | latent decoder maps denoised latent to image |
| Main comparison | Transformer denoiser vs U-Net denoiser |
| Scaling axis | depth, width, patch size, token count, Gflops |

## Latent Patch Tokenization

DiT operates in latent space. Given:

$$
z_t\in\mathbb{R}^{h\times w\times c},
$$

split it into patches of size $p\times p$:

$$
N
=
\frac{h w}{p^2}.
$$

Each patch is projected into a token:

$$
x_i = W_{\text{patch}}\operatorname{vec}(z_{t,i}) + b.
$$

The Transformer receives:

$$
X = [x_1,\dots,x_N] + P,
$$

where $P$ is positional information.

Smaller patch size gives more tokens:

$$
p\downarrow
\Rightarrow
N\uparrow
\Rightarrow
\text{more compute and potentially better quality}.
$$

## Transformer Denoiser

A DiT block is a Transformer block applied to latent patch tokens:

$$
H_{\ell+1}
=
\operatorname{Block}_{\ell}(H_\ell, t, y).
$$

The self-attention part follows the usual form:

$$
Q = H W_Q,
\qquad
K = H W_K,
\qquad
V = H W_V,
$$

$$
\operatorname{Attention}(H)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

The key change from U-Net diffusion is the spatial mixing operator:

| Backbone | Spatial Mixing |
| --- | --- |
| U-Net denoiser | convolution, down/up sampling, skip paths |
| DiT denoiser | global self-attention over latent patch tokens |

## Conditioning

Diffusion denoisers need timestep conditioning. Class-conditional image generation also needs label conditioning.

DiT studies conditioning mechanisms, with adaptive layer normalization as a central route. Abstractly:

$$
c = g(t,y),
$$

where $t$ is timestep and $y$ is optional class label.

The conditioned normalization can be written:

$$
\operatorname{adaLN}(h,c)
=
\gamma(c)\odot
\frac{h-\mu(h)}{\sigma(h)}
+
\beta(c).
$$

This lets the conditioning signal modulate the Transformer block without changing the token sequence length.

## Output Head

After Transformer blocks, tokens are mapped back to latent patches:

$$
\hat{\epsilon}_{i}
=
W_{\text{out}} h_i + b_{\text{out}}.
$$

The patches are unpatchified:

$$
\{\hat{\epsilon}_i\}_{i=1}^{N}
\rightarrow
\hat{\epsilon}_\theta(z_t,t,y)
\in
\mathbb{R}^{h\times w\times c}.
$$

The diffusion loss can use the standard noise-prediction form:

$$
\mathcal{L}
=
\mathbb{E}_{z_0,\epsilon,t,y}
\left[
\left\|
\epsilon
-
\epsilon_\theta(z_t,t,y)
\right\|_2^2
\right].
$$

## Scaling View

DiT evaluates scaling through forward-pass compute:

$$
\text{quality}
\approx
f(\text{Gflops}).
$$

Compute can increase by:

| Axis | Effect |
| --- | --- |
| depth | more Transformer blocks |
| width | larger hidden dimension |
| patch size $p$ | smaller $p$ gives more tokens |
| token count $N$ | attention and MLP cost increase |

The important empirical claim is that increasing DiT compute through these axes consistently improves FID in the tested ImageNet setting.

## Why It Matters

DiT is important because it moves the architecture conversation from:

$$
\text{diffusion model}
=
\text{U-Net denoiser}
$$

to:

$$
\text{diffusion model}
=
\text{objective}
+
\text{replaceable denoising backbone}.
$$

This opened the path for Transformer-heavy image/video diffusion backbones where scale, tokenization, and conditioning are first-class design variables.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet class-conditional generation | Transformer denoisers can outperform prior diffusion models under large scale |
| scaling curves | higher Gflops correlate with lower FID in the tested setup |
| patch-size comparisons | token count is a meaningful scaling route |
| conditioning ablations | block conditioning design matters for diffusion Transformers |

## Limits

- DiT is evaluated mainly in class-conditional ImageNet latent diffusion.
- Transformer attention cost grows with token count.
- The scaling claim is tied to compute, data, training recipe, and benchmark setting.
- U-Net backbones still remain strong in many settings, especially when locality and multi-scale skip structure are valuable.

## Diffusion interface

DiT changes the denoising backbone while preserving the latent diffusion interface. Let $z_0$ be a clean latent, $\epsilon\sim\mathcal{N}(0,I)$, and $\alpha_t,\sigma_t$ be a noise schedule. A noisy latent can be written:

$$
z_t
=
\alpha_t z_0
+
\sigma_t\epsilon.
$$

The denoiser receives $(z_t,t,c)$, where $c$ may be a class label or another condition, and predicts a target such as noise:

$$
\hat\epsilon
=
\epsilon_\theta(z_t,t,c).
$$

The standard training objective is:

$$
\mathcal{L}_{\epsilon}
=
\mathbb{E}_{z_0,\epsilon,t,c}
\left[
\left\|
\epsilon-\epsilon_\theta(z_t,t,c)
\right\|_2^2
\right].
$$

The architecture boundary is:

| Layer | DiT changes? | Contract |
| --- | --- | --- |
| data/latent encoder | usually inherited from latent diffusion | image to continuous latent |
| forward noising process | no | clean latent to noisy latent |
| denoising backbone | yes | noisy latent, timestep, condition to prediction |
| sampler | not fundamentally changed | predicted denoising target to next latent |
| latent decoder | usually inherited | final latent to image |

This prevents a common attribution error. A lower FID can result from the denoising backbone, latent autoencoder, sampler, guidance, training budget, or data recipe. The DiT paper primarily isolates the backbone and its compute scaling within the stated setup.

## From latent grid to token sequence

Let the noisy latent be:

$$
z_t\in\mathbb{R}^{h\times w\times c}.
$$

With patch side length $p$, the spatial token count is:

$$
N
=
\frac{h}{p}
\frac{w}{p}
=
\frac{hw}{p^2}.
$$

The $i$-th patch has shape $p\times p\times c$ and is flattened:

$$
u_i
=
\operatorname{vec}(z_{t,i})
\in
\mathbb{R}^{p^2c}.
$$

The patch projection is:

$$
x_i
=
u_i W_E+b_E,
\qquad
W_E\in\mathbb{R}^{p^2c\times d}.
$$

After adding positional information:

$$
H_0
=
[x_1,\ldots,x_N]
+
P,
\qquad
H_0\in\mathbb{R}^{N\times d}.
$$

The patch size is an architectural knob with two coupled effects:

| Patch size | Token count | Spatial detail | Attention cost |
| --- | --- | --- | --- |
| larger $p$ | lower | coarser | lower |
| smaller $p$ | higher | finer | higher |

This is not identical to changing image resolution. A fixed latent resolution and a changed patch size alter sequence length without changing the input image or diffusion schedule.

## DiT block

A simplified Transformer block is:

$$
\tilde H_\ell
=
H_\ell
+
\operatorname{MSA}
\left(
\operatorname{Norm}(H_\ell)
\right),
$$

$$
H_{\ell+1}
=
\tilde H_\ell
+
\operatorname{MLP}
\left(
\operatorname{Norm}(\tilde H_\ell)
\right).
$$

For diffusion, the block must also receive timestep and condition information. DiT uses adaptive layer normalization variants to modulate the normalized hidden state.

Let the condition embedding be:

$$
c
=
\operatorname{MLP}_{\mathrm{cond}}
\left(
e_t+e_y
\right),
$$

where $e_t$ is a timestep embedding and $e_y$ is an optional class embedding. The condition produces scale and shift vectors:

$$
(\gamma(c),\beta(c))
=
W_c c+b_c.
$$

Adaptive normalization is:

$$
\operatorname{adaLN}(h,c)
=
\gamma(c)\odot
\operatorname{LN}(h)
+
\beta(c).
$$

The conditioning interface changes the block function from:

$$
H_{\ell+1}=B_\ell(H_\ell)
$$

to:

$$
H_{\ell+1}=B_\ell(H_\ell,c).
$$

This allows all tokens to share the same global condition without appending a separate condition token to the sequence.

## adaLN-Zero

A residual block can be initialized so that its residual branch initially contributes almost nothing:

$$
H_{\ell+1}
=
H_\ell
+
\alpha(c)\odot
\operatorname{MSA}
\left(
\gamma_1(c)\odot\operatorname{LN}(H_\ell)
+
\beta_1(c)
\right),
$$

followed by a similarly modulated MLP residual branch.

With $\alpha(c)$ initialized near zero, the block starts close to an identity mapping:

$$
H_{\ell+1}\approx H_\ell
\quad\text{at initialization}.
$$

The condition-dependent gates are not only a way to inject labels. They also control how strongly each block's residual transformation is activated for a given timestep and class.

## Output head

The final hidden sequence:

$$
H_L\in\mathbb{R}^{N\times d}
$$

is projected back to patch-shaped predictions:

$$
\hat u_i
=
W_O h_i+b_O,
\qquad
\hat u_i\in\mathbb{R}^{p^2c_o}.
$$

The patches are reassembled:

$$
\{\hat u_i\}_{i=1}^{N}
\xrightarrow{\operatorname{unpatchify}}
\hat y_t
\in
\mathbb{R}^{h\times w\times c_o}.
$$

For a noise-prediction model, $c_o=c$. For a model that predicts variance or another parameterization, the output channel contract changes and must be recorded.

The model is structurally symmetric:

$$
\text{latent grid}
\rightarrow
\text{patch sequence}
\rightarrow
\text{Transformer}
\rightarrow
\text{latent grid}.
$$

Unlike a U-Net, there is no spatial downsampling and upsampling hierarchy inside the vanilla denoiser.

## U-Net versus DiT

The comparison should be made at the denoising backbone boundary.

| Property | U-Net denoiser | DiT denoiser |
| --- | --- | --- |
| primitive spatial operation | convolution | self-attention and MLP |
| locality | built in | learned through attention |
| multi-scale hierarchy | encoder-decoder with skip paths | not inherent in the vanilla block |
| token count | feature-map resolution at each stage | latent patch sequence |
| conditioning | feature injection, cross-attention, normalization | timestep/class modulation, optionally cross-attention |
| global interaction | indirect or multi-scale | direct among all tokens |
| scaling analysis | stage widths/resolutions | depth, width, patch size, Gflops |

DiT does not claim that convolution is unnecessary for every diffusion problem. It tests whether a standard Transformer can provide a scalable denoising backbone when the latent representation and compute are sufficiently large.

## Attention cost

For hidden width $d$ and $N$ latent patches, global self-attention has approximate score-matrix cost:

$$
O(N^2d).
$$

The tokenization relation gives:

$$
N=\frac{hw}{p^2},
\qquad
O(N^2d)
=
O\left(
\frac{h^2w^2}{p^4}d
\right).
$$

Halving the patch size increases token count by approximately four and the quadratic attention term by approximately sixteen, ignoring projection and implementation effects.

The MLP component has approximately:

$$
O(Nd d_{\mathrm{ff}})
$$

cost. At different token counts, attention and MLP may dominate in different regimes. Reporting only parameter count does not reveal actual denoising cost.

For video or 3D latent grids, token count also grows with time or depth:

$$
N_{\mathrm{video}}
=
\frac{t h w}{p_t p_h p_w}.
$$

The same DiT abstraction can scale poorly if the tokenization contract is not redesigned.

## Model variants and compute

DiT variants change depth, width, and patch size. A model name such as DiT-XL/2 encodes both backbone scale and patch size. The slash-two variant has a smaller patch size than a slash-eight variant, and therefore a longer token sequence at the same latent resolution.

The important comparison variables are:

| Variable | Interpretation |
| --- | --- |
| parameter count | capacity of the denoising network |
| Gflops per forward pass | approximate compute per denoising step |
| token count | spatial sequence length |
| denoising steps | number of network evaluations per sample |
| latent resolution | information available to the denoiser |
| training steps | optimization budget |

If a model has better FID but uses more denoising steps, larger latent resolution, or higher Gflops, the result is not a pure architecture comparison.

## Conditioning strategies

The base DiT setting uses class and timestep conditions. The same block can be extended for other conditions.

### Global modulation

Encode a condition $c$ and modulate each block:

$$
H_{\ell+1}=B_\ell(H_\ell,c).
$$

This is compact and keeps sequence length unchanged.

### Condition tokens

Append condition tokens:

$$
H_0=[x_1,\ldots,x_N,e_c]+P.
$$

This lets attention learn interactions between condition and spatial tokens, but increases sequence length.

### Cross-attention

Use a condition sequence $C$:

$$
\operatorname{CrossAttn}(H,C)
=
\operatorname{softmax}
\left(
\frac{(HW_Q)(CW_K)^\top}{\sqrt{d_k}}
\right)
CW_V.
$$

This is useful for text or structured conditioning but introduces additional memory and interface complexity.

The conditioning choice should be classified separately from the denoiser backbone. A Transformer with cross-attention and one with adaptive normalization may share the same spatial backbone while differing in condition injection.

## Scaling claim

The paper studies the relationship between model compute and generation quality:

$$
\operatorname{FID}
\approx
f(\operatorname{Gflops},\text{data},\text{training},\text{sampler}).
$$

Within the reported class-conditional ImageNet setting, larger DiT models and higher compute correlate with improved FID. The paper reports FID 2.27 for DiT-XL/2 on the 256x256 benchmark in the cited setup.

That number is evidence for a benchmark result, not a universal law:

$$
\text{observed scaling}
\ne
\text{architecture-independent scaling law}.
$$

To interpret the claim, compare the same latent autoencoder, training data, diffusion objective, sampler, guidance, and evaluation protocol. Report matched Gflops and either matched optimization budget or an explicit compute budget.

## Why DiT belongs in both architecture and generative shelves

The strongest claim is about the architecture:

$$
\text{U-Net denoiser}
\rightarrow
\text{Transformer denoiser}.
$$

The evidence is generated through a diffusion objective and sampling process. Therefore the canonical note belongs in Architecture Papers, while the generative-model shelf should link to it rather than duplicate it.

| Paper claim | Canonical shelf |
| --- | --- |
| replacing the denoising backbone | Architecture Papers |
| changing noise, score, or velocity objective | Generative Model Papers |
| improving sampler steps or ODE integration | Generative Model Papers |
| applying DiT to molecules or proteins | Computational Biology plus Generative Models if generation is central |
| reducing attention memory or kernel traffic | Systems or Architecture depending on strongest claim |

## Evidence and ablations

Useful DiT ablations separate backbone, conditioning, tokenization, and compute:

| Ablation | Question |
| --- | --- |
| U-Net versus Transformer | is the gain from denoiser family? |
| patch size | how does token count affect quality and cost? |
| depth and width | does more compute improve the same architecture family? |
| adaLN variant | how does conditioning injection affect optimization? |
| latent versus pixel input | how much comes from the representation interface? |
| sampler and steps | is the quality gain independent of inference budget? |
| same Gflops, different parameters | is parameter count or actual compute the better predictor? |
| same model, different data scale | does the weak inductive bias need more data? |

An ablation that changes multiple columns at once cannot establish an architecture claim. Scaling plots are most useful when read with the training and compute contract attached.

## Failure modes

### Token explosion

At high resolution or small patch size:

$$
N\propto\frac{1}{p^2},
\qquad
\text{attention memory}\propto N^2.
$$

This can make a small patch-size change dominate GPU memory and wall-clock time.

### Weak locality

A vanilla Transformer does not guarantee that nearby latent patches interact first. It may learn local structure from data, but optimization and data requirements can differ from a convolutional model.

### Latent bottleneck

If the latent autoencoder discards information, a larger DiT cannot recover it:

$$
\text{denoiser capacity}
\not\Rightarrow
\text{missing latent information}.
$$

The autoencoder and denoiser should be evaluated as separate architectural components.

### Conditioning leakage

If class or text information is exposed through an unintended path, conditional generation can look better without demonstrating stronger denoising structure.

### Compute confounding

FID can improve because of longer training, larger batch size, more sampler steps, or more total FLOPs. Report all of these before assigning credit to the Transformer backbone.

## Relation to nearby papers

| Paper | Main relation |
| --- | --- |
| [[papers/architectures/ddpm|DDPM]] | defines the denoising diffusion training family; DiT changes the backbone |
| [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]] | supplies the latent-space generation interface and conditioning context |
| [[papers/architectures/vision-transformer|Vision Transformer]] | supplies the patch-token and Transformer vision analogy |
| [[papers/architectures/taming-transformers|Taming Transformers]] | uses discrete image tokens and an autoregressive Transformer prior |
| [[papers/architectures/neural-discrete-representation-learning|VQ-VAE]] | learned discrete tokenizer; DiT usually operates on continuous latent grids |
| [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] | defines the Transformer attention backbone used as a denoiser primitive |

## Computational biology transfer

For a molecular or protein diffusion model, the DiT abstraction can be reused only after defining the tokenization and symmetry contract. A 3D structure may be represented as atom or residue tokens, graph nodes with edge attributes, continuous coordinate fields, or a latent grid learned by an autoencoder.

The vanilla image DiT does not automatically preserve geometric symmetries. For coordinates $X$, a model may need:

$$
f(RX+t)
=
Rf(X)+t
$$

for equivariant outputs, or:

$$
f(RX+t)
=
f(X)
$$

for invariant scalar outputs. A molecular DiT must therefore be routed through [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]] or the relevant geometry contract when the task requires it.

This is an example of reusing the Transformer denoiser pattern without copying the image inductive bias unchanged.

## Reproduction checklist

- [ ] Specify the latent autoencoder, latent resolution, and channel dimension.
- [ ] Record patch size, token count, hidden width, depth, and MLP expansion.
- [ ] State timestep embedding and class/condition injection mechanism.
- [ ] Report adaLN or adaLN-Zero details, including initialization.
- [ ] Define diffusion target: noise, velocity, $x_0$, or another parameterization.
- [ ] Match sampler, number of denoising steps, guidance, and evaluation seed policy.
- [ ] Compare U-Net and DiT under matched data, latent, training, and compute contracts.
- [ ] Report parameters, Gflops, attention memory, wall-clock time, and denoising steps.
- [ ] Track quality as a function of compute rather than reporting one model point only.
- [ ] For biological data, document tokenization, coordinate frame, symmetry, and geometry-preserving augmentations.

## Takeaway

DiT makes the denoising backbone a replaceable architecture:

$$
\boxed{
\text{noisy latent grid}
\rightarrow
\text{patch tokens}
\rightarrow
\text{conditioned Transformer}
\rightarrow
\text{denoising prediction}
}
$$

Its durable lesson is not simply that Transformers can generate images. It is that a generative objective can be paired with a general architecture family, then analyzed through tokenization, conditioning, compute, and scaling. The correct comparison is the complete denoising contract, not a bare claim that Transformer beats U-Net.

## Concepts

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]

## Related

- [[papers/architectures/ddpm|Denoising Diffusion Probabilistic Models]]
- [[papers/architectures/latent-diffusion-models|Latent Diffusion Models]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
