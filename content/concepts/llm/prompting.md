---
title: Prompting
tags:
  - llm
  - prompting
  - context
---

# Prompting

Prompting is the practice of specifying a language model task through instructions, context, examples, constraints, and output format. It is not a substitute for data, tools, or verification; it is the interface layer between a task and a model.

A prompt-conditioned generation can be written as:

$$
\hat{y}
\sim
p_\theta(y \mid p)
$$

where $p$ is the prompt and $y$ is the generated output.

A more realistic prompt decomposes into:

$$
p
=
(I, C, E, F)
$$

where $I$ is the instruction, $C$ is context, $E$ is evidence or examples, and $F$ is the requested format.

## Key Ideas

- Instructions define the task, but context and examples often dominate behavior.
- Prompting can control style, scope, constraints, and output shape.
- Ambiguous prompts produce ambiguous evaluation.
- Prompt quality should be judged by external task success, not by fluent answers.
- If correctness depends on private state, tools, or calculation, prompting alone is usually insufficient.

## Practical Checks

- What evidence is provided, and what evidence is missing?
- Are instructions and examples consistent?
- Is the output format machine-checkable?
- Are untrusted documents treated as data rather than instructions?
- Is there a verification step outside the model?

## Related

- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/verification/prompt-injection|Prompt injection]]
