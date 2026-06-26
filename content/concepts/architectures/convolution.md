---
title: Convolution
tags:
  - architectures
  - convolution
  - neural-networks
---

# Convolution

Convolution applies a local filter across positions with shared weights. It is the core operation behind many CNNs, image models, audio models, contact-map models, and voxel-grid models.

For a 1D signal:

$$
y_i = \sum_{r=-R}^{R} w_r x_{i+r}
$$

$w_r$ is the filter weight at offset $r$, and the same weights are reused at every position $i$.

For a 2D input with channels:

$$
y_{u,v,c_{\mathrm{out}}}
=
\sum_{\Delta u,\Delta v,c_{\mathrm{in}}}
W_{\Delta u,\Delta v,c_{\mathrm{in}},c_{\mathrm{out}}}
x_{u+\Delta u,v+\Delta v,c_{\mathrm{in}}}
+ b_{c_{\mathrm{out}}}
$$

The filter mixes local spatial neighborhoods and input channels.

## Locality and Equivariance

Before boundary effects, convolution is translation equivariant:

$$
\operatorname{Conv}(T_a x)
=
T_a\operatorname{Conv}(x)
$$

$T_a$ shifts the input by offset $a$. This means shifting the input shifts the output in the same way.

## Design Parameters

- Kernel size: local window size.
- Stride: step size between filter positions.
- Padding: boundary handling.
- Dilation: spacing between kernel positions.
- Groups: whether channels are split into independent convolution groups.
- Dimension: 1D sequence, 2D image/contact map, or 3D voxel/grid.

## Checks

- What axis is spatial, sequence, channel, batch, or feature?
- Does padding introduce artifacts at boundaries?
- Does stride or pooling discard resolution needed by the task?
- Is a dense grid representation appropriate, or is the object naturally a graph or point cloud?
- Does the convolution encode the right symmetry for the data?

## Related

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
