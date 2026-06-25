---
title: Language Model
tags:
  - llm
  - generative-models
---

# Language Model

A language model assigns probabilities to token sequences and can generate text by predicting the next token from previous context.

An autoregressive language model factorizes sequence probability as:

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

The training objective is usually next-token negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

## Uses

- Text generation and summarization.
- Classification or extraction through prompting.
- Tool-call generation in agents.
- Code and documentation assistance.
- Retrieval synthesis when paired with external evidence.

## Checks

- Is the model answering from context, parameters, retrieval, or tools?
- Is the output grounded in verifiable evidence?
- Are constraints, format, and refusal boundaries explicit?
- Is the task better handled by a structured tool than free-form generation?

## Related

- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[agents/tool-use|Tool use]]
