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

## Checks

- What frame rate and clip length are used?
- Is sampling uniform, event-centered, or learned?
- Does the model need short motion, long-range temporal reasoning, or both?
- Are adjacent clips from the same source split across train and test?
- Are audio and text aligned to the correct temporal segment?

## Related

- [[concepts/modalities/image|Image]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
