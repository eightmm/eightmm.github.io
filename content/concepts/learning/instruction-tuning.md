---
title: Instruction Tuning
tags:
  - instruction-tuning
  - fine-tuning
  - language-models
---

# Instruction Tuning

Instruction tuning adapts a pretrained model to follow natural-language or structured task instructions. It is usually supervised fine-tuning on examples of instructions and desired responses.

Given examples $(u_i, r_i)$, where $u_i$ is an instruction or conversation context and $r_i$ is the target response, the objective is:

$$
\theta^\*
=
\arg\min_\theta
-\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(r_i\mid u_i)
$$

For token sequences:

$$
\log p_\theta(r_i\mid u_i)
=
\sum_{t=1}^{T_i}
\log p_\theta(r_{i,t}\mid u_i,r_{i,<t})
$$

## Why It Matters

- Converts a pretrained next-token model into a more usable assistant or task solver.
- Teaches formatting, refusal style, tool-use patterns, and task conventions.
- Often precedes preference optimization or reinforcement learning from feedback.
- Can overfit to prompt style instead of learning robust task behavior.

## Data Questions

- Are instructions diverse or template-heavy?
- Are answers verified, preference-ranked, synthetic, or human-written?
- Are tool calls, citations, and structured outputs represented consistently?
- Are unsafe, private, or unverifiable examples filtered?

## Checks

- Does evaluation distinguish instruction following from factual correctness?
- Are examples leaking benchmark prompts or answers?
- Does tuning degrade base capabilities such as coding, reasoning, or domain knowledge?
- Are responses judged by exact match, execution, human preference, or verifier checks?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/core/agent-loop|Agent loop]]
