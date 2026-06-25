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

## Key Ideas

- Convolutions reuse the same filter across positions, giving parameter sharing and local pattern detection.
- Kernel size, stride, dilation, and padding define the receptive field and output resolution.
- Stacking layers grows the effective context while preserving a useful locality bias.
- Pooling or strided convolution trades spatial detail for larger context and cheaper computation.
- CNNs can process sequences, contact maps, grids, images, and voxelized molecular structures when the input has a stable neighborhood layout.

## Practical Checks

- Verify the input layout: channel-first, channel-last, 1D, 2D, or 3D.
- Check whether boundary padding changes the meaning of edge positions.
- For structural biology inputs, ask what information was lost when coordinates became grids or voxels.
- Compare CNNs against [[concepts/architectures/gnn|GNNs]] when the object is naturally a graph rather than a dense grid.

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/learning/index|Learning methods]]
- [[research/structure-based-ai/index|Structure-based AI]]
