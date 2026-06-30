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

## Common Decoders

| Decoder | Rule | Use when | Risk |
| --- | --- | --- | --- |
| Greedy | choose $\arg\max$ each step | deterministic extraction, simple constrained output | local optimum, repetition |
| Temperature sampling | sample from softened logits | brainstorming or diverse text | variance and unsupported claims |
| Top-$k$ | sample from $k$ highest-probability tokens | limit tail noise | fixed $k$ ignores distribution shape |
| Nucleus top-$p$ | sample from smallest set with cumulative mass $p$ | adaptive diversity | unstable with poor calibration |
| Beam search | keep multiple high-probability partial sequences | translation or short sequence search | generic output, length bias |

Top-$k$ defines a candidate set:

$$
S_k
=
\operatorname{TopK}_{x} z_x
$$

Nucleus sampling defines the smallest set whose probability mass exceeds $p$:

$$
S_p
=
\min_{S}
\left\{
S:
\sum_{x\in S} p(x\mid x_{<t}) \ge p
\right\}
$$

Sampling then renormalizes over the chosen set.

## Length and Stop Rules

Decoding is also controlled by maximum tokens and stop criteria.

$$
\operatorname{stop}
\iff
x_t \in \mathcal{S}_{\mathrm{stop}}
\lor
t \ge T_{\max}
\lor
\operatorname{valid}(x_{\le t})=1
$$

| Control | Purpose | Failure mode |
| --- | --- | --- |
| max tokens | bound cost and context use | truncates reasoning or JSON |
| stop sequence | end at known delimiter | stops inside quoted text |
| schema validation | accept only valid structure | valid syntax with wrong content |
| repetition penalty | reduce loops | can distort technical wording |
| multiple samples | expose uncertainty | increases verification cost |

## Decoding vs Verification

Decoding parameters affect variability, but they do not verify truth.

| Task | Typical decoding | Still needs |
| --- | --- | --- |
| JSON extraction | low temperature, schema constraint | semantic validation against source |
| factual answer | low temperature or deterministic | citation and source support |
| code generation | low-to-moderate sampling | tests and review |
| brainstorming | higher diversity | filtering and human selection |
| agent tool call | constrained output | tool result handling and side-effect check |

## Practical Checks

- Is the task deterministic extraction or creative generation?
- Are temperature, top-$k$, top-$p$, max tokens, and stop rules documented?
- Does the decoder preserve required structure or schema?
- Are multiple samples needed to estimate variability?
- Is the selected answer verified after decoding?
- Is output variability part of the claim, or just hidden noise?
- Are stop rules tested against the intended output format?

## Related

- [[concepts/llm/language-model|Language model]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
