---
title: Video
tags:
  - modalities
  - video
  - vision
---

# Video

Video is a time-ordered sequence of image frames:

$$
X \in \mathbb{R}^{T \times H \times W \times C}
$$

where $T$ is the number of frames. Video modeling adds temporal structure on top of image modeling.

## Common Strategies

- Frame encoder plus temporal pooling.
- 3D convolution over space and time.
- Frame or patch tokens processed by [[concepts/architectures/transformer|Transformers]].
- Recurrent or [[concepts/architectures/state-space-model|state-space]] temporal models.
- Multimodal alignment with audio, subtitles, or text descriptions.

## Sampling Boundary

Video models rarely consume every frame. A clip sampler defines the actual input:

$$
X_{\mathrm{clip}}
=
\operatorname{sample}
(X_{\mathrm{video}};
t_0,
T,
\Delta t)
$$

where $t_0$ is the start time, $T$ is clip length, and $\Delta t$ is the frame stride. Event-centered sampling, random clips, sliding windows, and key-frame sampling test different abilities.

## Temporal Output Spaces

Video outputs can be clip-level, frame-level, event-level, or sequence-level:

$$
Y
\in
\{\text{clip label},
\text{temporal segment},
\text{frame mask},
\text{caption},
\text{answer},
\text{future sequence}\}
$$

The split and metric should match the output. A clip-level label does not prove temporal localization unless the evaluation asks for event boundaries.

## Leakage Risks

- Adjacent clips from the same source video appear in different splits.
- Scene, camera, uploader, or timestamp identifies the label.
- Subtitles, audio, or metadata reveal answers when the intended task is visual.
- Future frames are visible in a causal prediction task.

## Checks

- What frame rate and clip length are used?
- Is sampling uniform, event-centered, or learned?
- Does the model need short motion, long-range temporal reasoning, or both?
- Are adjacent clips from the same source split across train and test?
- Are audio and text aligned to the correct temporal segment?
- Is the task allowed to use future frames?
- Are labels clip-level, frame-level, or event-boundary labels?
- Are long videos evaluated without leaking neighboring windows?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/tasks/time-series-forecasting|Time-series forecasting]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
