---
title: Sequence-to-Sequence
tags:
  - tasks
  - sequence-modeling
  - structured-prediction
---

# Sequence-to-Sequence

Sequence-to-sequence is a task family where both input and output are sequences, and the output length may differ from the input length.

The contract is:

$$
x_{1:T_x}
\rightarrow
y_{1:T_y}.
$$

Examples include translation, summarization, speech recognition, captioning, code generation from text, and sequence-conditioned protein or molecule generation.

## Autoregressive Form

A common factorization is:

$$
p(y_{1:T_y}\mid x_{1:T_x})
=
\prod_{t=1}^{T_y}
p(y_t\mid y_{<t},x_{1:T_x}).
$$

The model must define:

- how the input sequence is represented;
- how previous output tokens are used;
- when generation stops;
- what output tokens are valid;
- which metric evaluates the sequence.

## Architecture Patterns

| Pattern | Reading |
| --- | --- |
| encoder-decoder RNN | encode input, decode output autoregressively |
| attention encoder-decoder | decoder dynamically reads source states |
| Transformer encoder-decoder | self-attention over source and target plus cross-attention |
| decoder-only prompting | concatenate input and output into one causal sequence |

## Evaluation Boundary

Sequence-level success is not always token-level success.

| Metric Type | Measures | Risk |
| --- | --- | --- |
| token loss | next-token prediction | exposure bias and sequence-level mismatch |
| exact match | full output correctness | harsh for semantically equivalent outputs |
| BLEU/ROUGE | n-gram overlap | weak semantic validity |
| task-specific validity | output satisfies constraints | may ignore fluency |
| downstream utility | output works in a later pipeline | evaluator can be expensive or biased |

## Checks

- Is the output length fixed or variable?
- Is decoding greedy, beam search, sampling, or constrained?
- Is teacher forcing used during training?
- Does evaluation use token metrics or sequence-level metrics?
- Is the model allowed to attend to the whole input?
- Are source padding and target causal masks correct?

## Related

- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/rnn-encoder-decoder|RNN Encoder-Decoder]]
- [[papers/architectures/neural-machine-translation-align-translate|Bahdanau attention]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
