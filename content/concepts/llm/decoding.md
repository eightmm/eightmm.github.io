---
title: Decoding
tags:
  - llm
  - generation
  - decoding
---

# Decoding

Decoding is the procedure used to turn next-token probabilities into an output sequence. The model gives a distribution; the decoding rule chooses or samples tokens from it.

At step $t$, a language model predicts:

$$
p_\theta(x_t \mid x_{<t})
$$

Greedy decoding chooses the most likely token:

$$
\hat{x}_t
=
\arg\max_x
p_\theta(x \mid x_{<t})
$$

Sampling draws from a temperature-scaled distribution:

$$
p_T(x)
=
\frac{\exp(z_x/T)}
{\sum_{x'}\exp(z_{x'}/T)}
$$

where $z_x$ is the token logit and $T$ is temperature.

## Key Ideas

- Greedy decoding is deterministic but can be brittle or repetitive.
- Higher temperature increases randomness; lower temperature makes outputs more concentrated.
- Top-$k$ and nucleus sampling restrict the candidate token set before sampling.
- Beam search explores multiple high-probability sequences but can favor generic outputs.
- Decoding changes output behavior without changing model weights.

## Practical Checks

- Is the task deterministic extraction or creative generation?
- Are temperature, top-$k$, top-$p$, max tokens, and stop rules documented?
- Does the decoder preserve required structure or schema?
- Are multiple samples needed to estimate variability?
- Is the selected answer verified after decoding?

## Related

- [[concepts/llm/language-model|Language model]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
