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

## Output Space

Segmentation output is dense:

$$
f_\theta(x)
\rightarrow
\hat{P}
\in
[0,1]^{C\times H\times W}
$$

where $\hat{P}_{c,h,w}$ is the predicted probability for class $c$ at position $(h,w)$. The hard mask is usually:

$$
\hat{Y}_{h,w}
=
\arg\max_c
\hat{P}_{c,h,w}
$$

For multi-label masks, each class may instead use an independent threshold.

## Loss Choices

Common losses include pixel cross-entropy:

$$
\mathcal{L}_{\mathrm{CE}}
=
-
\sum_{h,w}
\log
\hat{P}_{Y_{h,w},h,w}
$$

and Dice-style overlap:

$$
\operatorname{Dice}
=
\frac{
2|\hat{Y}\cap Y|
}{
|\hat{Y}|+|Y|
}
$$

The loss should match whether missing boundaries, small objects, foreground coverage, or class balance matters most.

## Failure Boundary

Segmentation errors can come from:

- boundary shift;
- missing small regions;
- false positive regions;
- merged or split instances;
- label ambiguity from weak masks;
- resizing or interpolation artifacts.

## Checks

- Are boundaries or interior regions more important?
- Is class imbalance severe?
- Are masks generated manually, automatically, or from weak labels?
- Does resizing preserve thin structures?
- For scientific data, does the mask correspond to a physically meaningful region?
- Are unlabeled pixels ignored, treated as background, or treated as unknown?
- Is the task semantic, instance, panoptic, or multi-label segmentation?
- Is metric computed per image, per class, or over the whole dataset?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/localization|Localization]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/architectures/feature-pyramid-networks|Feature Pyramid Networks]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
- [[papers/architectures/segment-anything|Segment Anything]]
