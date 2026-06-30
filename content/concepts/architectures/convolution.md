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

## Cross-Correlation vs Convolution

Deep learning libraries usually implement cross-correlation rather than mathematically flipped convolution:

$$
y_i
=
\sum_{r=-R}^{R}
w_r x_{i+r}
$$

instead of

$$
y_i
=
\sum_{r=-R}^{R}
w_r x_{i-r}
$$

The distinction rarely matters for learning because $w$ is learned, but it matters when comparing formulas with signal-processing texts.

## Locality and Equivariance

Before boundary effects, convolution is translation equivariant:

$$
\operatorname{Conv}(T_a x)
=
T_a\operatorname{Conv}(x)
$$

$T_a$ shifts the input by offset $a$. This means shifting the input shifts the output in the same way.

## Shape Formula

For one spatial dimension, output length is:

$$
L_{\mathrm{out}}
=
\left\lfloor
\frac{L_{\mathrm{in}} + 2P - D(K-1) - 1}{S}
+ 1
\right\rfloor
$$

where $K$ is kernel size, $P$ padding, $D$ dilation, and $S$ stride.

| Parameter | Effect | Tradeoff |
| --- | --- | --- |
| kernel size $K$ | local receptive field | larger kernels cost more and smooth locality |
| stride $S$ | downsampling | loses resolution but reduces compute |
| padding $P$ | boundary treatment | can create artificial edge context |
| dilation $D$ | sparse larger field | can miss fine local patterns |
| groups | channel factorization | cheaper but less cross-channel mixing |

## Receptive Field

Stacked convolutions increase receptive field. For stride-1 layers with kernel size $K$, a rough receptive field after $L$ layers is:

$$
R_L
=
1 + L(K-1)
$$

Pooling, stride, and dilation change this faster. The key question is whether the receptive field covers the dependency scale required by the task.

| Data | Useful convolution view |
| --- | --- |
| image | local texture and spatial hierarchy |
| audio | local temporal waveform pattern |
| sequence | local motif or n-gram-like feature |
| contact map | local 2D pattern over residue pairs |
| voxel grid | local 3D occupancy or density pattern |

## Convolution vs Attention

| Property | Convolution | Attention |
| --- | --- | --- |
| connectivity | local by default | global or masked |
| weight sharing | fixed offsets share weights | content-dependent weights |
| inductive bias | locality and translation equivariance | flexible pairwise interaction |
| compute pattern | dense local stencil | pairwise token interaction |
| good baseline for | images, grids, local signals | long-range dependencies and variable relations |

This is why convolution remains useful even when Transformers dominate many sequence tasks: it encodes a strong local prior cheaply.

## Long Convolution

For sequence models, convolution does not have to be local. A long convolution can use a kernel over the full context:

$$
y_t
=
\sum_{\tau=0}^{t}
k_\tau x_{t-\tau}.
$$

If the kernel is implicitly parameterized,

$$
k_\tau=f_\theta(\tau),
$$

the model can represent long filters without storing an independent parameter for every offset. [[papers/architectures/hyena|Hyena]] is the canonical paper note here for gated implicit long convolution as a dense-attention-free sequence mixer.

Exponential moving average is a related sequence-memory pattern:

$$
z_t
=
\alpha x_t
+
(1-\alpha)z_{t-1}
=
\sum_{i=1}^{t}
\alpha(1-\alpha)^{t-i}x_i.
$$

It behaves like a decayed causal filter. [[papers/architectures/mega|Mega]] uses this moving-average bias inside a gated attention block rather than as a standalone convolutional sequence model.

## Design Parameters

- Kernel size: local window size.
- Stride: step size between filter positions.
- Padding: boundary handling.
- Dilation: spacing between kernel positions.
- Groups: whether channels are split into independent convolution groups.
- Dimension: 1D sequence, 2D image/contact map, or 3D voxel/grid.
- Receptive field: whether stacked layers can cover the dependency scale.
- Boundary policy: whether padding is physically meaningful for the data.

## Checks

- What axis is spatial, sequence, channel, batch, or feature?
- Does padding introduce artifacts at boundaries?
- Does stride or pooling discard resolution needed by the task?
- Is a dense grid representation appropriate, or is the object naturally a graph or point cloud?
- Does the convolution encode the right symmetry for the data?
- Is the receptive field large enough without destroying resolution?
- Are local interactions enough, or is attention/graph message passing needed?

## Canonical Papers

| Paper | Why It Matters |
| --- | --- |
| [WaveNet](/papers/architectures/wavenet) | dilated causal convolution for autoregressive sequence generation |
| [Mega](/papers/architectures/mega) | exponential moving average as local sequence memory inside gated attention |
| [Hyena](/papers/architectures/hyena) | implicit long convolution plus gating for long-context sequence modeling |
| [ConvNeXt](/papers/architectures/convnext) | depthwise convolution as modern vision token mixing |

## Related

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/mega|Mega]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
