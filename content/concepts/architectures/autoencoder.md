---
title: Autoencoder
tags:
  - architectures
  - representation-learning
---

# Autoencoder

An autoencoder learns an encoder and decoder that reconstruct an input through a bottleneck representation. It is both an architecture pattern and a representation-learning objective.

The basic form is:

$$
z = f_\theta(x)
$$

$$
\hat{x} = g_\phi(z)
$$

$$
\mathcal{L}_{\mathrm{rec}} = d(x,\hat{x})
$$

where $f_\theta$ is the encoder, $g_\phi$ is the decoder, $z$ is the latent representation, and $d$ is a reconstruction loss.

For continuous inputs, a common reconstruction loss is:

$$
\mathcal{L}_{\mathrm{MSE}}
=
\lVert x-\hat{x}\rVert_2^2
$$

For categorical tokens, reconstruction is often a negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-\sum_{i\in \mathcal{M}} \log p_\phi(x_i \mid z)
$$

where $\mathcal{M}$ can be all positions or only masked or corrupted positions.

## Key Ideas

- The bottleneck forces the model to compress information needed for reconstruction.
- The encoder and decoder can be MLPs, CNNs, Transformers, GNNs, or geometry-aware modules.
- Denoising autoencoders reconstruct clean inputs from corrupted inputs.
- Variational autoencoders add a probabilistic latent variable and a regularization term.
- Masked autoencoding is closely related to [[concepts/learning/self-supervised-learning|self-supervised learning]].

## Variants

| Variant | Objective | Main use |
| --- | --- | --- |
| bottleneck autoencoder | reconstruct through low-dimensional $z$ | compression, representation learning |
| denoising autoencoder | reconstruct $x$ from corrupted $\tilde{x}$ | robust features, score-like intuition |
| sparse autoencoder | add sparsity penalty on $z$ | interpretable or disentangled features |
| masked autoencoder | reconstruct hidden tokens or patches | self-supervised pretraining |
| variational autoencoder | reconstruction plus latent regularization | probabilistic latent generation |

For a denoising autoencoder:

$$
\tilde{x}\sim q(\tilde{x}\mid x),
\qquad
z=f_\theta(\tilde{x}),
\qquad
\hat{x}=g_\phi(z)
$$

The corruption process $q(\tilde{x}\mid x)$ is part of the learning signal, not a harmless detail.

## Representation Boundary

Good reconstruction does not automatically imply a useful representation. A model can reconstruct low-level detail while ignoring the factor needed for a downstream task.

| Claim | Need to check |
| --- | --- |
| compression works | reconstruction loss, bitrate or latent size, failure cases |
| representation transfers | frozen probe, fine-tuning protocol, split rule |
| generation works | sample validity, likelihood or downstream utility |
| denoising helps | corruption type, clean target, robustness metric |

## Practical Checks

- Check what reconstruction target is used: raw input, masked tokens, coordinates, features, or graph attributes.
- Verify whether good reconstruction implies useful downstream representation for the target task.
- Watch for trivial identity mappings when the bottleneck or corruption is too weak.
- Separate architecture choices from the reconstruction objective.
- State whether $z$ is deterministic, stochastic, discrete, continuous, global, or token-wise.
- Check that train/validation preprocessing and masking policies match the reported claim.

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/generative-models/vae|Variational autoencoder]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/u-net|U-Net]]
