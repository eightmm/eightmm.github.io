---
title: U-Net
tags:
  - architectures
  - cnn
  - generative-models
---

# U-Net

U-Net is an encoder-decoder architecture with skip connections between matching resolutions. It is common in segmentation, image-to-image models, and diffusion backbones.

The high-level form is:

$$
h_{\ell+1}^{\mathrm{down}} = D_\ell(h_\ell^{\mathrm{down}})
$$

$$
h_{\ell}^{\mathrm{up}} = U_\ell\left(
\operatorname{Concat}(h_{\ell+1}^{\mathrm{up}}, h_\ell^{\mathrm{down}})
\right)
$$

Skip connections preserve high-resolution information while the bottleneck captures global context.

For a denoising model, the U-Net often predicts noise or a velocity field:

$$
\epsilon_\theta(x_t, t, c)
\quad\text{or}\quad
v_\theta(x_t, t, c)
$$

where $x_t$ is a noisy input, $t$ is the noise level or time, and $c$ is optional conditioning. Conditioning can enter through time embeddings, class embeddings, cross-attention, or feature-wise modulation.

## Shape Pattern

- Down path: reduce spatial resolution while increasing channels.
- Bottleneck: mix broad context at low resolution.
- Up path: recover resolution using learned upsampling or transposed convolution.
- Skip paths: copy high-resolution features from encoder to decoder.

The architecture is strongest when input and output share a spatial grid, such as segmentation masks, denoised images, voxel maps, contact maps, or latent feature grids.

## Why It Matters

- Useful when output has the same spatial format as input.
- Preserves local detail through skip connections.
- Widely used as a denoising network in [[concepts/generative-models/diffusion-model|diffusion models]].

## Checks

- Are skip connections concatenated or added?
- Does the model operate on images, voxels, contact maps, or latent grids?
- Is conditioning injected at the bottleneck, every block, or through cross-attention?

## Related

- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/tasks/segmentation|Segmentation]]
