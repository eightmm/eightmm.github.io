---
title: Object Detection
tags:
  - tasks
  - vision
  - detection
---

# Object Detection

Object detection predicts both object categories and object locations. The output is usually a set of labeled bounding boxes:

$$
\hat{Y} = \{(\hat{c}_i, \hat{b}_i, \hat{s}_i)\}_{i=1}^{N}
$$

where $\hat{c}_i$ is a class, $\hat{b}_i$ is a bounding box, and $\hat{s}_i$ is a confidence score.

## Box Representation

A box can be represented by corners:

$$
b = (x_{\min}, y_{\min}, x_{\max}, y_{\max})
$$

or by center, width, and height:

$$
b = (x_c, y_c, w, h)
$$

Intersection-over-union compares predicted and ground-truth boxes:

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

## Checks

- Are labels object-level, image-level, or weakly supervised?
- Does the dataset contain small objects, occlusion, dense objects, or class imbalance?
- What IoU threshold defines a correct detection?
- Are augmented boxes transformed correctly with the image?
- Does non-maximum suppression remove true neighboring objects?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
