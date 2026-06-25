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

## Checks

- Are examples representative of the desired behavior?
- Are instructions consistent with examples?
- Are labels or answers leaked into evaluation prompts?
- Is the behavior stable under reordered or paraphrased examples?
- Would fine-tuning or a tool be more reliable than prompting?

## Related

- [[concepts/llm/context-window|Context window]]
- [[agents/context-engineering|Context engineering]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/leakage|Leakage]]
