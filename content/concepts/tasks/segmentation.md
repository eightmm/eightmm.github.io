---
title: Segmentation
tags:
  - tasks
  - vision
  - segmentation
---

# Segmentation

Segmentation assigns labels to pixels, voxels, residues, atoms, or regions. It is used when localization must be more precise than a bounding box.

For semantic image segmentation, the model predicts:

$$
\hat{Y} \in \{1,\ldots,C\}^{H \times W}
$$

where each pixel receives one of $C$ class labels.

## Types

- Semantic segmentation: class label per position.
- Instance segmentation: separate mask per object instance.
- Panoptic segmentation: semantic and instance segmentation combined.
- 3D segmentation: voxel, point-cloud, or structure-region labels.

## Metrics

Intersection-over-union for a class $c$ is:

$$
\operatorname{IoU}_c =
\frac{|\hat{Y}_c \cap Y_c|}{|\hat{Y}_c \cup Y_c|}
$$

Mean IoU averages this across classes:

$$
\operatorname{mIoU}=\frac{1}{C}\sum_{c=1}^{C}\operatorname{IoU}_c
$$

## Checks

- Are boundaries or interior regions more important?
- Is class imbalance severe?
- Are masks generated manually, automatically, or from weak labels?
- Does resizing preserve thin structures?
- For scientific data, does the mask correspond to a physically meaningful region?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/evaluation/metric|Metric]]
