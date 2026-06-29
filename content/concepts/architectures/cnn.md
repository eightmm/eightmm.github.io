---
title: Convolutional Neural Networks
tags:
  - architectures
  - cnn
  - convolution
---

# Convolutional Neural Networks

Convolutional neural networks use local filters with shared weights. They are useful when neighboring positions carry repeated local patterns, such as images, grids, sequences, or voxelized structures.

For a 1D signal, a convolution can be written as:

$$
y_i = \sum_{r=-R}^{R} w_r x_{i+r}
$$

The same weights $w_r$ are reused at every position $i$, which gives parameter sharing and locality.

For images or grids with channels, a 2D convolution is:

$$
y_{u,v,c_{\mathrm{out}}}
= \sum_{\Delta u,\Delta v,c_{\mathrm{in}}}
W_{\Delta u,\Delta v,c_{\mathrm{in}},c_{\mathrm{out}}}
x_{u+\Delta u,v+\Delta v,c_{\mathrm{in}}}
+ b_{c_{\mathrm{out}}}
$$

Convolution is translation equivariant before boundary effects and pooling:

$$
\operatorname{Conv}(T_a x) = T_a\operatorname{Conv}(x)
$$

where $T_a$ shifts the input by offset $a$.

## Key Ideas

- Convolutions reuse the same filter across positions, giving parameter sharing and local pattern detection.
- Kernel size, stride, dilation, and padding define the receptive field and output resolution.
- Stacking layers grows the effective context while preserving a useful locality bias.
- Pooling or strided convolution trades spatial detail for larger context and cheaper computation.
- CNNs can process sequences, contact maps, grids, images, and voxelized molecular structures when the input has a stable neighborhood layout.

## Receptive Field

The receptive field is the part of the input that can affect one output position. For stride 1 convolutions with kernel size $k$ and $L$ layers, a simple receptive field approximation is:

$$
R_L = 1 + L(k-1)
$$

With dilation $d_\ell$ at layer $\ell$, the effective kernel span grows:

$$
R_L
= 1 + \sum_{\ell=1}^{L} (k_\ell - 1)
\prod_{j<\ell} s_j d_\ell
$$

where $s_j$ is the stride before layer $\ell$. This matters because many CNN claims are really claims about whether the relevant dependency is local enough for the chosen receptive field.

## CNN vs Transformer vs GNN

| Question | CNN | Transformer | GNN |
| --- | --- | --- | --- |
| Input layout | dense grid or regular sequence | token sequence or set with positional information | explicit graph with nodes and edges |
| Main bias | locality and translation sharing | global content-based interaction | neighborhood message passing |
| Long-range relation | grows through depth, dilation, pooling | direct attention path | grows through message-passing depth or graph edges |
| Cost driver | resolution, channels, kernel size | token count and attention pattern | nodes, edges, message functions |
| Common failure | too-small receptive field or grid artifact | quadratic cost or positional shortcut | bad graph construction or over-smoothing |

For molecular or protein work, the choice often depends on the representation:

| Representation | CNN fit |
| --- | --- |
| image, contact map, distance map | strong fit if local grid patterns matter |
| voxelized 3D structure | possible, but resolution and rotation handling matter |
| molecular graph | usually prefer [[concepts/architectures/gnn|GNN]] or graph transformer |
| raw sequence | 1D CNN can be a baseline, but long-range dependency may need attention or SSM |

## Design Knobs

| Knob | Effect | Risk |
| --- | --- | --- |
| kernel size | local context per layer | large kernels increase cost and may blur locality |
| stride | downsampling and cheaper later layers | can discard small features |
| dilation | larger context without pooling | gridding artifacts if poorly chosen |
| padding | output shape and edge behavior | boundary positions may get artificial context |
| channel width | feature capacity | memory and overfitting |
| pooling | invariance and lower resolution | loses exact position information |

## Paper Reading Checks

| Claim | Check |
| --- | --- |
| better local feature extractor | kernel size, depth, receptive field, and augmentation |
| better image or grid model | resolution, preprocessing, and boundary handling |
| efficient model | wall time, memory, resolution, and batch size |
| structure-aware CNN | whether rotation, translation, and coordinate frame issues are handled |
| better than GNN | whether graph construction or voxelization changes the information available |

## Practical Checks

- Verify the input layout: channel-first, channel-last, 1D, 2D, or 3D.
- Check whether boundary padding changes the meaning of edge positions.
- For structural biology inputs, ask what information was lost when coordinates became grids or voxels.
- Compare CNNs against [[concepts/architectures/gnn|GNNs]] when the object is naturally a graph rather than a dense grid.

## Related

- [[concepts/architectures/convolution|Convolution]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/learning/index|Learning methods]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
