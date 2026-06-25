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

## Checks

- What sample rate, window size, and hop length are used?
- Is loudness or channel normalization applied consistently?
- Are background noise and recording devices confounded with labels?
- Is the target speech recognition, audio classification, event detection, or generation?
- If transcripts are used, are they aligned and separated from evaluation labels?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/video|Video]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
