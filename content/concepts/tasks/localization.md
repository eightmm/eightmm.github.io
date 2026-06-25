---
title: Localization
tags:
  - tasks
  - localization
  - spatial
---

# Localization

Localization predicts where an entity, event, region, or signal is located. It sits between global prediction and dense [[concepts/tasks/segmentation|segmentation]]: the output is spatial, but not always a full mask.

The output can be written as:

$$
\hat{Y}
=
\{(\hat{c}_i,\hat{z}_i,\hat{s}_i)\}_{i=1}^{N}
$$

where $\hat{c}_i$ is an optional class, $\hat{z}_i$ is a location object, and $\hat{s}_i$ is a confidence score.

## Output Types

- Point: $(x,y)$, $(x,y,z)$, residue index, timestamp, or token span.
- Box: $(x_{\min},y_{\min},x_{\max},y_{\max})$.
- Region proposal: coarse spatial region before segmentation.
- Keypoint set: joints, landmarks, atoms, residues, or anchor points.
- Span: start/end positions in text, sequence, audio, or video.
- Transform: relative pose, rotation, translation, or alignment.

## Loss And Metric

For point localization:

$$
\mathcal{L}_{\mathrm{point}}
=
\lVert \hat{z} - z \rVert_2^2
$$

For boxes, overlap is often measured by:

$$
\operatorname{IoU}(A,B)
=
\frac{|A\cap B|}{|A\cup B|}
$$

For sequence spans:

$$
\operatorname{overlap}
=
\frac{|[\hat{a},\hat{b}]\cap[a,b]|}{|[\hat{a},\hat{b}]\cup[a,b]|}
$$

The metric should match the output type. A coordinate error, IoU, span F1, and top-k hit rate are not interchangeable.

## Failure Modes

- Predicting the right class at the wrong location.
- Predicting a plausible location that is unavailable at deployment time.
- Learning annotation bias instead of the physical or semantic site.
- Collapsing multiple valid locations into one averaged point.
- Evaluating with a tolerance that is too loose for the downstream decision.

## Checks

- What spatial object is predicted: point, box, keypoint, span, transform, or region?
- Is location measured in pixels, tokens, residues, atoms, coordinates, frames, or time?
- What tolerance defines a correct localization?
- Are augmentations applied consistently to labels?
- Does preprocessing leak the true location, pose, or reference frame?

## Related

- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/coordinate-prediction|Coordinate prediction]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/evaluation/metric-selection|Metric selection]]
