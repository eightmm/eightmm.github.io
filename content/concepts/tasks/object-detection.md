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

## Matching Problem

Detection is a set prediction task. Predictions must be matched to ground-truth objects before scoring:

$$
\operatorname{match}(\hat{b}_i,b_j)
=
\mathbf{1}
\left[
\operatorname{IoU}(\hat{b}_i,b_j)
\ge
\tau
\right]
$$

where $\tau$ is an IoU threshold. A prediction can be wrong because the class is wrong, the box is poorly localized, the confidence score is badly calibrated, or the object is duplicated or missed.

## Training Target

Many detectors combine classification and localization losses:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda
\mathcal{L}_{\mathrm{box}}
+
\gamma
\mathcal{L}_{\mathrm{obj}}
$$

where $\mathcal{L}_{\mathrm{obj}}$ may represent objectness or foreground/background confidence depending on the detector family.

## Evaluation Boundary

Average precision evaluates ranked detections:

$$
\operatorname{AP}
=
\int_0^1
p(r)\,dr
$$

where $p(r)$ is precision as a function of recall. AP depends on confidence scores, matching threshold, class imbalance, and duplicate suppression. Report the IoU threshold or threshold range.

## Checks

- Are labels object-level, image-level, or weakly supervised?
- Does the dataset contain small objects, occlusion, dense objects, or class imbalance?
- What IoU threshold defines a correct detection?
- Are augmented boxes transformed correctly with the image?
- Does non-maximum suppression remove true neighboring objects?
- Are confidence scores calibrated enough for the downstream threshold?
- Are duplicate detections, missed objects, and localization errors separated in error analysis?
- Is evaluation per class, macro-averaged, micro-averaged, or weighted by frequency?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/tasks/localization|Localization]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/faster-r-cnn|Faster R-CNN]]
- [[papers/architectures/mask-r-cnn|Mask R-CNN]]
- [[papers/architectures/detr|DETR]]
