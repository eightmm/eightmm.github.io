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

The training objective is usually next-token [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]]:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

## Token Distribution

At each step, the model produces logits over a vocabulary:

$$
z_t
=
f_\theta(x_{<t})
\in
\mathbb{R}^{|\mathcal{V}|}
$$

The next-token distribution is:

$$
p_\theta(x_t=v\mid x_{<t})
=
\frac{\exp(z_{t,v})}
{\sum_{u\in\mathcal{V}}\exp(z_{t,u})}
$$

Generation then samples or selects from this distribution using a [[concepts/llm/decoding|decoding]] rule.

## Conditioning Sources

An LLM answer can come from several sources:

$$
p(y\mid q)
\approx
p_\theta
\left(
y
\mid
\text{instructions},
\text{context},
\text{retrieved evidence},
\text{tool results},
q
\right)
$$

The model does not inherently know which source is authoritative. That ordering must be supplied by prompt design, tool contracts, retrieval policy, and verification.

## Parametric vs External Knowledge

Parametric knowledge is stored in model weights. External knowledge is supplied at inference time through context, retrieval, tools, or files. For public wiki workflows, external evidence is usually preferable when facts can change:

$$
\text{current answer}
\Rightarrow
\text{current source}
\Rightarrow
\text{verified citation or artifact}
$$

This is especially important for software versions, APIs, papers, benchmarks, and deployment behavior.

## Uses

- Text generation and summarization.
- Classification or extraction through prompting.
- Tool-call generation in agents.
- Code and documentation assistance.
- Retrieval synthesis when paired with external evidence.

## Failure Modes

- Hallucination: plausible text not supported by evidence.
- Prompt sensitivity: small wording changes alter behavior.
- Context loss: important evidence is absent or truncated.
- Recency error: parametric knowledge is stale.
- Tool misuse: the model writes an invalid tool call or misreads tool output.
- Overconfident uncertainty: probability-like language does not mean calibrated confidence.

## Checks

- Is the model answering from context, parameters, retrieval, or tools?
- Is the output grounded in verifiable evidence?
- Are constraints, format, and refusal boundaries explicit?
- Is the task better handled by a structured tool than free-form generation?
- Is the answer sensitive to decoding parameters or prompt examples?
- Does the task need current information, citations, or direct source inspection?

## Related

- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[agents/tools/tool-use|Tool use]]
