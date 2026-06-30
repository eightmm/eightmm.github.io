---
title: Sequence to Sequence Learning with Neural Networks
aliases:
  - papers/seq2seq
  - papers/sequence-to-sequence-learning
tags:
  - papers
  - architectures
  - recurrent
  - sequence-modeling
  - machine-translation
---

# Sequence to Sequence Learning with Neural Networks

> The paper made sequence-to-sequence learning a practical end-to-end neural architecture by using deep LSTMs to encode a variable-length input sequence and decode a variable-length output sequence.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Sequence to Sequence Learning with Neural Networks |
| Authors | Ilya Sutskever, Oriol Vinyals, Quoc V. Le |
| Year | 2014 |
| Venue | NeurIPS 2014 |
| arXiv | [1409.3215](https://arxiv.org/abs/1409.3215) |
| NeurIPS | [proceedings page](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks) |
| Status | full note started |

## Question

Before attention became standard, the key architecture question was:

$$
\text{Can a neural network map one variable-length sequence to another variable-length sequence end-to-end?}
$$

For machine translation, the input and output lengths differ:

$$
x_{1:T_x}
\rightarrow
y_{1:T_y}.
$$

A fixed-size classifier cannot directly express this mapping. The paper asks whether a deep recurrent encoder-decoder can handle it by compressing the source sentence into a vector and then generating the target sentence autoregressively.

This sits between [[papers/architectures/long-short-term-memory|LSTM]] as a memory cell and [[papers/architectures/neural-machine-translation-align-translate|Bahdanau attention]] as the next step that removes the single-vector bottleneck.

## Main Claim

The main architecture claim is:

$$
\operatorname{Encoder}_{\text{LSTM}}(x_{1:T_x}) \to c,
\qquad
\operatorname{Decoder}_{\text{LSTM}}(c, y_{<t}) \to p(y_t\mid y_{<t}, x_{1:T_x}).
$$

In words: one deep LSTM reads the source sequence into a fixed-dimensional vector, and another deep LSTM decodes a target sequence from that vector.

The target probability is factorized left to right:

$$
p(y_{1:T_y}\mid x_{1:T_x})
=
\prod_{t=1}^{T_y}
p(y_t\mid y_{<t}, c).
$$

where:

| Symbol | Meaning |
| --- | --- |
| $x_{1:T_x}$ | source token sequence |
| $y_{1:T_y}$ | target token sequence |
| $c$ | fixed-dimensional sentence representation from the encoder |
| $p(y_t\mid y_{<t},c)$ | decoder next-token distribution |

The durable contribution is not the exact BLEU number. It is the encoder-decoder contract for variable-length sequence transduction.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | variable-length source sentence $x_{1:T_x}$ |
| Output | variable-length target sentence $y_{1:T_y}$ |
| Encoder | multi-layer LSTM over source tokens |
| Bridge | final encoder state used as a fixed-length sentence vector |
| Decoder | multi-layer LSTM language model conditioned on the encoder vector |
| Generation | autoregressive next-token prediction |
| Training objective | maximum likelihood / cross-entropy over target tokens |
| Core bottleneck | the full source sequence must fit into one vector |
| Important trick | reverse the source token order to shorten dependency paths |

The architecture can be read as three blocks:

$$
E_\theta : x_{1:T_x} \mapsto c
$$

$$
D_\phi : (c, y_{<t}) \mapsto s_t
$$

$$
O_\psi : s_t \mapsto p(y_t\mid y_{<t},c).
$$

This pattern later survives in [[concepts/architectures/encoder-decoder|encoder-decoder]] Transformers, but the bridge changes from one vector to cross-attention over all encoder states.

## Encoder

The encoder is an LSTM recurrence over the source tokens:

$$
(h_t, m_t)
=
\operatorname{LSTM}_{\text{enc}}(x_t, h_{t-1}, m_{t-1}),
\qquad
t=1,\ldots,T_x.
$$

Here $h_t$ is the exposed hidden state and $m_t$ is the LSTM memory cell. A simplified sentence vector is:

$$
c = h_{T_x}
$$

or, in stacked LSTMs, the final hidden and cell states across layers initialize the decoder.

The important compression is:

$$
x_{1:T_x}
\rightarrow
c \in \mathbb{R}^d.
$$

No matter how long the source sentence is, the decoder initially receives a fixed-size summary.

## Decoder

The decoder is an autoregressive LSTM:

$$
(s_t, n_t)
=
\operatorname{LSTM}_{\text{dec}}(y_{t-1}, s_{t-1}, n_{t-1}; c).
$$

The output logits are:

$$
o_t = W_o s_t + b_o
$$

and the next-token distribution is:

$$
p(y_t=k\mid y_{<t},c)
=
\frac{\exp(o_{t,k})}{\sum_{j=1}^{V}\exp(o_{t,j})}.
$$

where $V$ is the target vocabulary size.

The training loss for one target sentence is:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T_y}
\log p(y_t^\ast\mid y_{<t}^\ast,c).
$$

This is teacher-forced maximum likelihood: during training, the decoder conditions on the gold previous target token $y_{t-1}^\ast$.

## Why Source Reversal Helps

The paper found that reversing source sentence order improves optimization.

Without reversal:

$$
x_1, x_2, \ldots, x_{T_x}
\rightarrow
y_1, y_2, \ldots, y_{T_y}.
$$

With reversal:

$$
x_{T_x}, x_{T_x-1}, \ldots, x_1
\rightarrow
y_1, y_2, \ldots, y_{T_y}.
$$

The intuition is dependency path length. In many translation pairs, early target words align with early source words. If the source is read in normal order, information about $x_1$ must survive until the final encoder state and then pass into early decoder steps:

$$
x_1 \to h_1 \to \cdots \to h_{T_x}=c \to s_1 \to y_1.
$$

If the source is reversed, the source word that often aligns with $y_1$ is closer to the final encoder state:

$$
x_1 \text{ is read near the end of the encoder}.
$$

This does not change model expressivity. It changes the optimization geometry by shortening many effective credit-assignment paths.

## Relation To Cho Encoder-Decoder

Both this paper and [[papers/architectures/rnn-encoder-decoder|Cho et al. RNN Encoder-Decoder]] use the same high-level decomposition:

$$
x_{1:T_x} \to c \to y_{1:T_y}.
$$

The difference is emphasis:

| Paper | Unit | Scope | Main Role |
| --- | --- | --- | --- |
| Cho et al. 2014 | gated recurrent unit / phrase encoder-decoder | phrase scoring in SMT | compact gated recurrence and phrase representation |
| Sutskever et al. 2014 | deep LSTM encoder-decoder | full sentence translation | practical end-to-end seq2seq translation |

Together they define the pre-attention seq2seq baseline.

## Relation To Bahdanau Attention

The limitation of this paper is the fixed vector:

$$
c=h_{T_x}.
$$

[[papers/architectures/neural-machine-translation-align-translate|Bahdanau attention]] changes the bridge from one vector to a sequence of encoder annotations:

$$
H=(h_1,\ldots,h_{T_x})
$$

and computes a decoder-step-specific context:

$$
c_t
=
\sum_{i=1}^{T_x}
\alpha_{t,i}h_i.
$$

The architectural move is:

| Fixed-vector seq2seq | Attention seq2seq |
| --- | --- |
| one context vector for the whole sentence | one context vector per output step |
| all source information compressed into $c$ | decoder can query source positions |
| source reversal helps optimization | learned alignment handles reordering more directly |
| bottleneck grows with task difficulty | memory grows with source length |

This makes the seq2seq paper important historically: it shows the contract, and attention shows where that contract was too narrow.

## Evidence To Read Carefully

The paper reports strong English-to-French translation results on WMT'14 using a deep LSTM system, including a standalone neural translation system and reranking of phrase-based SMT candidates.

For architecture reading, separate three claims:

| Claim | What Supports It | What It Does Not Prove Alone |
| --- | --- | --- |
| Variable-length seq2seq is trainable | end-to-end LSTM translation works competitively | that fixed-vector bottlenecks are optimal |
| Deep LSTMs can store sentence-level information | translation quality and learned phrase/sentence representations | that long-context memory is solved generally |
| Source reversal improves optimization | performance improves after reversing source order | that word order can be ignored |

The reported result is not just an architecture result. It also depends on data, tokenization, decoding, model size, training setup, and reranking protocol.

## Failure Modes

| Failure Mode | Mechanism | Later Response |
| --- | --- | --- |
| Fixed-vector bottleneck | all source information compressed into $c$ | attention and cross-attention expose all source states |
| Long-range dependency burden | early source information must survive many recurrent steps | LSTM gates, source reversal, attention |
| Sequential training/inference | recurrence prevents full time-axis parallelism | Transformer self-attention parallelizes token mixing |
| Exposure bias | training conditions on gold previous tokens, generation conditions on model outputs | scheduled sampling, sequence-level objectives, better decoding |
| Search dependence | translation quality depends on beam search and reranking | stronger decoders and direct generation pipelines |

## Why It Still Matters

This paper is worth keeping in an architecture shelf because it defines an interface that still appears everywhere:

$$
\text{encode input}
\rightarrow
\text{conditioned decoder}
\rightarrow
\text{sequence output}.
$$

Examples:

- machine translation: source sentence to target sentence;
- captioning: image representation to caption tokens;
- speech recognition: acoustic sequence to text sequence;
- code generation: prompt/context to output tokens;
- molecular design: condition representation to generated string, graph, or coordinate trajectory.

Even when the recurrent LSTM is replaced by a Transformer, diffusion backbone, or graph network, the conditional generation contract remains useful.

## Practical Checks

When reading a later seq2seq paper, check:

| Question | Why It Matters |
| --- | --- |
| What is the encoder state exposed to the decoder? | determines the information bottleneck |
| Is the decoder autoregressive, parallel, or iterative? | determines factorization and inference cost |
| Does the model use teacher forcing? | affects train-test mismatch |
| Is alignment learned explicitly, implicitly, or not at all? | affects long input handling |
| Are gains from architecture, decoding, data, or reranking? | prevents over-crediting the block |

## Where It Fits

| Axis | Placement |
| --- | --- |
| Architecture family | recurrent encoder-decoder |
| Core inductive bias | ordered causal recurrence with compressed memory |
| Main concept | [sequence-to-sequence](/concepts/tasks/sequence-to-sequence) |
| Predecessor | [LSTM](/papers/architectures/long-short-term-memory) |
| Parallel line | [RNN Encoder-Decoder](/papers/architectures/rnn-encoder-decoder) |
| Successor | [Bahdanau attention](/papers/architectures/neural-machine-translation-align-translate) |
| Later replacement | [Transformer](/papers/architectures/attention-is-all-you-need) |

## Related

- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/sequence-to-sequence|Sequence-to-sequence]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/rnn-encoder-decoder|Learning Phrase Representations using RNN Encoder-Decoder]]
- [[papers/architectures/neural-machine-translation-align-translate|Neural Machine Translation by Jointly Learning to Align and Translate]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
