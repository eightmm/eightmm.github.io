---
title: Audio
tags:
  - modalities
  - audio
  - signal-processing
---

# Audio

Audio is a continuous time signal usually sampled into a waveform:

$$
x_{1:T} \in \mathbb{R}^{T}
$$

Many models transform waveforms into time-frequency features. A common representation is the spectrogram:

$$
S_{t,f} = |\operatorname{STFT}(x)_{t,f}|^2
$$

where $t$ indexes time windows and $f$ indexes frequency bins.

## Common Representations

- Raw waveform samples.
- Spectrogram or log-mel spectrogram.
- Learned acoustic tokens.
- Aligned transcript text.
- Encoder embeddings from a pretrained speech or audio model.

## Windowing Boundary

Audio preprocessing usually segments the signal:

$$
x_{t:t+L}
=
(x_t,\ldots,x_{t+L-1})
$$

and a spectrogram uses overlapping windows:

$$
S
=
\operatorname{Spectrogram}
(x;\ \text{window},\ \text{hop},\ \text{sample rate})
$$

Window length controls temporal resolution, while frequency bins control spectral resolution. Changing these settings changes the input representation.

## Task Coupling

Audio tasks differ in output granularity:

$$
Y
\in
\{\text{clip label},
\text{event segment},
\text{transcript},
\text{speaker label},
\text{embedding},
\text{generated waveform}\}
$$

Speech recognition, sound event detection, speaker identification, and audio captioning should not share metrics without stating the output space.

## Leakage Risks

- Same speaker, recording session, device, room, or background appears across splits.
- Transcript text leaks class labels or downstream answers.
- Silence trimming or event-centered crops use label timing unavailable at deployment.
- Loudness, microphone, or compression artifacts become the shortcut.

## Checks

- What sample rate, window size, and hop length are used?
- Is loudness or channel normalization applied consistently?
- Are background noise and recording devices confounded with labels?
- Is the target speech recognition, audio classification, event detection, or generation?
- If transcripts are used, are they aligned and separated from evaluation labels?
- Is the split grouped by speaker, recording session, source, or environment when needed?
- Are event timestamps available at inference time?
- Does augmentation preserve timing, speaker identity, and label semantics?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/video|Video]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/anomaly-detection|Anomaly detection]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
