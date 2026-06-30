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

## Prompt Contract

A useful prompt separates task, context, evidence, constraints, and verifier.

$$
p
=
(I,\ C,\ E,\ R,\ F,\ V)
$$

| Part | Meaning | Good prompt question |
| --- | --- | --- |
| $I$ | instruction | What should be done? |
| $C$ | context | What state or background is available? |
| $E$ | evidence/examples | What should the model ground on or imitate? |
| $R$ | restrictions | What must not be used, changed, or exposed? |
| $F$ | format | What shape should the output take? |
| $V$ | verifier | How will correctness be checked? |

This is why a prompt can improve interface quality without replacing [[agents/verification/index|verification]].

## Prompt Types

| Type | Use for | Risk |
| --- | --- | --- |
| Direct instruction | simple known task | underspecified output |
| Few-shot examples | format or pattern transfer | examples overfit or conflict |
| Evidence-grounded prompt | answer from supplied sources | sources may be stale or adversarial |
| Structured-output prompt | JSON, table, schema, checklist | syntax valid but semantics wrong |
| Tool-use prompt | ask model to select action | side effects need external verifier |
| Critique prompt | review or find gaps | can invent issues without evidence |

## Prompting vs Context Engineering

Prompting writes the immediate instruction. [[agents/core/context-engineering|Context engineering]] decides what evidence and state enter the context at all.

| Question | Belongs closer to |
| --- | --- |
| What wording should the instruction use? | Prompting |
| Which files, chunks, or examples are included? | Context engineering |
| How is context ordered under token budget? | [[concepts/llm/context-packing|Context packing]] |
| How is output checked after generation? | [[agents/verification/index|Agent Verification]] |
| What tool may be called? | [[agents/tools/tool-use|Tool use]] |

## Evaluation Pattern

Prompt changes should be evaluated against a task set, not one pleasing answer.

$$
\Delta
=
\operatorname{score}(p_{\mathrm{new}}, \mathcal{T})
-
\operatorname{score}(p_{\mathrm{base}}, \mathcal{T})
$$

where $\mathcal{T}$ is a fixed set of tasks or examples.

| Evaluation item | Why |
| --- | --- |
| baseline prompt | prevents subjective improvement claims |
| fixed task set | makes before/after comparable |
| output rubric | separates style, correctness, safety, completeness |
| failure slices | shows which tasks got worse |
| external verifier | avoids judging fluency as correctness |

## Practical Checks

- What evidence is provided, and what evidence is missing?
- Are instructions and examples consistent?
- Is the output format machine-checkable?
- Are untrusted documents treated as data rather than instructions?
- Is there a verification step outside the model?
- Is the prompt being asked to solve a retrieval, calculation, or tool problem that should be externalized?
- Is the prompt change evaluated on repeated tasks, not a single anecdote?

## Related

- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/verification/index|Agent Verification]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/prompt-injection|Prompt injection]]
