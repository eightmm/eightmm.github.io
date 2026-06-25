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

## Key Ideas

- The bottleneck forces the model to compress information needed for reconstruction.
- The encoder and decoder can be MLPs, CNNs, Transformers, GNNs, or geometry-aware modules.
- Denoising autoencoders reconstruct clean inputs from corrupted inputs.
- Variational autoencoders add a probabilistic latent variable and a regularization term.
- Masked autoencoding is closely related to [[concepts/learning/self-supervised-learning|self-supervised learning]].

## Practical Checks

- Check what reconstruction target is used: raw input, masked tokens, coordinates, features, or graph attributes.
- Verify whether good reconstruction implies useful downstream representation for the target task.
- Watch for trivial identity mappings when the bottleneck or corruption is too weak.
- Separate architecture choices from the reconstruction objective.

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/generative-models/vae|Variational autoencoder]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/u-net|U-Net]]
