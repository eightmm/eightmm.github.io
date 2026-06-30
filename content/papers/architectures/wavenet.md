---
title: WaveNet
aliases:
  - papers/wavenet
  - papers/wavenet-generative-model-raw-audio
  - papers/a-generative-model-for-raw-audio
tags:
  - papers
  - architectures
  - audio
  - generative-models
---

# WaveNet

> The paper introduced a deep autoregressive model for raw audio using causal dilated convolutions, gated residual blocks, and sample-level likelihood training.

## Metadata

| Field | Value |
| --- | --- |
| Paper | WaveNet: A Generative Model for Raw Audio |
| Authors | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu |
| Year | 2016 |
| Venue | arXiv; ISCA Speech Synthesis Workshop demo |
| arXiv | [1609.03499](https://arxiv.org/abs/1609.03499) |
| DeepMind | [WaveNet blog](https://deepmind.google/blog/wavenet-a-generative-model-for-raw-audio/) |
| Status | full note started |

## One-Line Takeaway

WaveNet models raw audio autoregressively:

$$
p(\mathbf{x})
=
\prod_{t=1}^{T}
p(x_t\mid x_1,\ldots,x_{t-1}),
$$

using a stack of causal dilated convolutions so each output sample has a large receptive field over previous samples.

## Question

Audio is a high-rate sequence. At 16 kHz, one second has:

$$
16000
$$

samples. Directly modeling raw waveform samples requires both:

- causal generation, because future samples are unavailable;
- a long receptive field, because speech and music depend on structure over many time scales.

The architecture question is:

$$
\text{How can a neural network model raw waveform samples with long context while remaining trainable in parallel?}
$$

WaveNet answers with dilated causal convolutions.

## Main Claim

A deep causal convolutional network with exponentially increasing dilation can model raw audio waveforms and generate high-quality speech/music samples.

The core architecture:

$$
\text{causal convolution}
+
\text{dilation schedule}
+
\text{gated residual blocks}
\Rightarrow
\text{autoregressive raw-audio model}.
$$

This is important beyond audio because it shows how convolution can provide long-range sequence context without recurrence.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | discrete raw audio samples |
| Output | distribution over next audio sample |
| Factorization | autoregressive sample-level likelihood |
| Causality | output at time $t$ only depends on samples before $t$ |
| Context mechanism | dilated causal convolutions |
| Block | gated activation with residual and skip connections |
| Conditioning | optional local linguistic features or global speaker identity |
| Main task | raw audio generation, especially text-to-speech |
| Main bottleneck | sequential sampling at inference |

## Autoregressive Audio Likelihood

For waveform:

$$
\mathbf{x}=(x_1,\ldots,x_T),
$$

WaveNet factorizes:

$$
p(\mathbf{x})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t}).
$$

The training objective is negative log likelihood:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}, c),
$$

where $c$ is optional conditioning.

The model is trained with teacher forcing: previous ground-truth samples are available during training, so all positions can be evaluated in parallel through convolution.

## Causal Convolution

A causal convolution for output time $t$ only reads current and past inputs:

$$
y_t
=
\sum_{i=0}^{k-1}
w_i x_{t-i}.
$$

It must not read:

$$
x_{t+1},x_{t+2},\ldots
$$

because generation proceeds one sample at a time.

This causal contract is the sequence analogue of masking in a decoder-only Transformer:

$$
\text{no future information}.
$$

## Dilated Convolution

A dilated convolution skips input positions by a dilation factor $d$:

$$
y_t
=
\sum_{i=0}^{k-1}
w_i x_{t-di}.
$$

For kernel size $k=2$:

$$
y_t=w_0x_t+w_1x_{t-d}.
$$

If dilation doubles by layer:

$$
d_\ell=2^\ell,
$$

the receptive field grows exponentially with depth.

For a stack of $L$ layers with kernel size $k$, the receptive field is roughly:

$$
R
=
1
+
(k-1)
\sum_{\ell=0}^{L-1}
d_\ell.
$$

With doubling dilation:

$$
R
=
1
+
(k-1)(2^L-1).
$$

This is why WaveNet can cover long audio context with a moderate number of layers.

## Gated Residual Block

WaveNet uses gated activation units:

$$
z
=
\tanh(W_f * x)
\odot
\sigma(W_g * x),
$$

where:

| Symbol | Meaning |
| --- | --- |
| $*$ | dilated causal convolution |
| $W_f$ | filter convolution |
| $W_g$ | gate convolution |
| $\sigma$ | sigmoid gate |
| $\odot$ | elementwise product |

The block then uses residual and skip paths:

$$
x_{\ell+1}
=
x_\ell
+
R_\ell z_\ell,
$$

and:

$$
s
=
\sum_{\ell}
S_\ell z_\ell.
$$

The skip sum feeds the output network. Residual paths stabilize deep stacks, while skip paths expose features from multiple receptive-field scales.

## Output Distribution

The original WaveNet discretizes audio samples and predicts a categorical distribution.

For quantized sample value:

$$
x_t\in\{1,\ldots,K\},
$$

the output is:

$$
p_\theta(x_t=k\mid x_{<t},c)
=
\operatorname{softmax}(o_t)_k.
$$

The loss is cross-entropy over sample values:

$$
\mathcal{L}_t
=
-
\log p_\theta(x_t\mid x_{<t},c).
$$

Later neural vocoders and audio models often changed the output distribution, but the architecture idea of causal dilated waveform modeling remained influential.

## Conditioning

WaveNet supports conditioning:

$$
p(x_t\mid x_{<t},c).
$$

Global conditioning can represent speaker identity:

$$
c=g_{\text{speaker}}.
$$

Local conditioning can represent time-aligned linguistic or acoustic features:

$$
c_t
$$

that are injected into layers.

The paper's text-to-speech setting is therefore:

$$
\text{text or linguistic features}
\rightarrow
\text{conditioning}
\rightarrow
\text{raw waveform generation}.
$$

## Training Vs Sampling

Training can evaluate all time positions in parallel because ground-truth history is available:

$$
\{x_{<t}\}_{t=1}^T
$$

comes from data.

Sampling is sequential:

$$
x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_T.
$$

Each new sample becomes input for the next step.

This creates the major tradeoff:

| Phase | Property |
| --- | --- |
| training | parallel convolution over full sequence |
| sampling | sequential autoregressive generation |
| quality | high sample-level fidelity |
| latency | expensive without caching/distillation |

## Relation To CNNs And Transformers

WaveNet is a sequence CNN:

$$
\text{local filters}
+
\text{dilation}
\rightarrow
\text{large causal context}.
$$

Compared with a Transformer:

| Axis | WaveNet | Transformer |
| --- | --- | --- |
| mixing | convolutional filters | attention |
| context growth | dilation stack | direct token-token attention |
| causality | causal convolution | causal mask |
| train parallelism | yes | yes |
| sampling | sequential | sequential for autoregressive decoding |
| inductive bias | local translation-in-time filters | content-based global interaction |

WaveNet remains useful when reading architectures that prefer convolutional locality and long receptive fields over attention.

## Relation To State-Space And Long-Sequence Models

WaveNet, [[papers/architectures/transformer-xl|Transformer-XL]], and [[papers/architectures/mamba|Mamba]] all address long sequence context differently:

| Model | Context Mechanism |
| --- | --- |
| WaveNet | dilated causal convolution receptive field |
| Transformer-XL | cached hidden-state memory plus relative attention |
| Mamba | selective state-space recurrence |

This gives a clean reading axis:

$$
\text{long context}
\neq
\text{one architecture family}.
$$

The question is whether the task benefits more from locality, content-based lookup, or recurrent state.

## Why It Belongs In Architecture Papers

WaveNet is a canonical architecture paper because it popularized:

- direct raw waveform modeling;
- causal convolution for autoregressive generation;
- dilated convolution for exponential receptive-field growth;
- gated residual blocks for deep sequence CNNs;
- conditioning for controllable audio generation.

The architecture lesson applies outside speech:

$$
\text{dilation}
\Rightarrow
\text{large receptive field without dense attention}.
$$

## Evidence Pattern

The paper supports the architecture with:

| Evidence | What It Supports |
| --- | --- |
| text-to-speech listening tests | raw waveform generation improves naturalness |
| speaker conditioning | one model can represent multiple speakers |
| music modeling samples | architecture is not limited to speech |
| phoneme recognition experiment | learned waveform features are useful discriminatively |
| qualitative audio | sample fidelity is central for audio claims |

For audio generation, subjective evaluation matters because likelihood alone may not capture perceptual quality.

## Practical Reading Checks

| Question | Why |
| --- | --- |
| What is the sample rate? | determines sequence length and required context |
| What is the receptive field? | limits temporal dependencies |
| What output distribution is used? | affects audio quality and likelihood |
| Is generation autoregressive? | determines latency |
| Is conditioning local or global? | controls TTS and speaker behavior |
| Are listening tests reported? | perceptual quality is central |
| Is inference accelerated? | raw autoregressive audio is slow |

## Limits

- Autoregressive sampling is slow for long audio.
- Receptive field is finite.
- Discretized output can introduce modeling choices.
- Audio quality depends heavily on conditioning and data quality.
- Likelihood and human perception may disagree.
- Later systems often need distillation, flow, diffusion, or non-autoregressive vocoders for speed.

The concise limitation:

$$
\text{high-fidelity autoregressive audio}
\neq
\text{fast audio generation}.
$$

## What To Remember

- WaveNet models raw audio sample by sample.
- Causal convolutions prevent future leakage.
- Dilated convolutions grow receptive field exponentially.
- Gated residual blocks stabilize deep waveform modeling.
- Training is parallel with teacher forcing; generation is sequential.
- The paper is a key reference for convolutional long-context sequence modeling.

## Links

- [[concepts/modalities/audio|Audio]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
