---
title: In-Context Learning
tags:
  - llm
  - prompting
---

# In-Context Learning

In-context learning is when a model adapts behavior from examples, instructions, or evidence placed in the context window without changing model weights.

A prompt with examples can be viewed as:

$$
y^\*
=
\arg\max_y
p_\theta
\left(
y \mid
I,\,
(x_1,y_1),\ldots,(x_k,y_k),\,
x^\*
\right)
$$

where $I$ is an instruction and $(x_i,y_i)$ are examples.

## What Changes

In-context learning changes the conditional distribution, not the model parameters:

$$
\theta
\ \text{fixed},
\qquad
p_\theta(y\mid x)
\rightarrow
p_\theta(y\mid I, E, x)
$$

where $E$ is a set of examples or evidence. This is different from [[concepts/learning/fine-tuning|fine-tuning]], where training updates $\theta$.

## Example Selection

Example quality matters. A small demonstration set can be selected by similarity:

$$
E_k(x^\*)
=
\operatorname{topk}_{(x_i,y_i)}
\operatorname{sim}(x^\*,x_i)
$$

or by coverage of expected cases:

$$
E
=
E_{\mathrm{easy}}
\cup
E_{\mathrm{edge}}
\cup
E_{\mathrm{failure}}
$$

Random examples, mislabeled examples, or examples with hidden leakage can make in-context learning look stronger than it is.

## Instruction and Demonstration Conflict

If instructions and examples disagree, the model may follow either:

$$
I \not\equiv E
\Rightarrow
\text{unstable behavior}
$$

For workflows that require reliability, examples should illustrate the instruction rather than silently redefining it.

## Evaluation Risk

Few-shot prompts can leak labels if examples are drawn from the evaluation set or from near-duplicates:

$$
(x^\*,y^\*)\in E
\Rightarrow
\text{invalid evaluation}
$$

For retrieval-augmented workflows, retrieved examples should be checked against the test unit and split unit.

## Checks

- Are examples representative of the desired behavior?
- Are instructions consistent with examples?
- Are labels or answers leaked into evaluation prompts?
- Is the behavior stable under reordered or paraphrased examples?
- Would fine-tuning or a tool be more reliable than prompting?
- Are examples selected before seeing target labels?
- Are examples current, public-safe, and free of private data?
- Is the prompt tested on edge cases, not only happy-path examples?
- Is the output format enforced with [[concepts/llm/structured-output|structured output]] when downstream tools consume it?

## Related

- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/context-engineering|Context engineering]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
