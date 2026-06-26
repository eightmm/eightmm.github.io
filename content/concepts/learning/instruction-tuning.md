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

## Data Contract

| Field | Question |
| --- | --- |
| Instruction source | human-written, synthetic, benchmark-derived, conversation logs, or tool traces? |
| Response source | expert answer, model answer, verified output, execution result, or preference winner? |
| Task mixture | coding, QA, math, tool use, domain science, safety, formatting? |
| Target format | free text, JSON, function call, citation, proof, code, or command? |
| Filtering | private, unsafe, unverifiable, duplicate, or benchmark-leaking examples removed? |
| Loss mask | are prompt tokens, tool outputs, and response tokens masked correctly? |

For supervised instruction tuning:

$$
\mathcal{L}_{\mathrm{SFT}}
=
-
\sum_{t\in \mathcal{M}}
\log p_\theta(r_t\mid u,r_{<t})
$$

where $\mathcal{M}$ is the set of response tokens that contribute to the loss. Loss masking is part of the training contract.

## Capability Boundary

Instruction tuning can teach response format without proving domain correctness:

| Claim | Needs |
| --- | --- |
| follows instructions | held-out instruction-following eval |
| uses tools correctly | executable tool-call or environment evaluation |
| improves reasoning | reasoning benchmark with contamination check |
| improves domain science | domain task metric and expert/error analysis |
| improves safety | adversarial and policy-specific safety eval |

Formatting compliance and factual correctness should be evaluated separately.

## Failure Modes

| Failure | Symptom |
| --- | --- |
| benchmark contamination | training examples overlap with eval prompts |
| style overfitting | model mimics answer style but not task skill |
| catastrophic forgetting | base capabilities degrade after tuning |
| tool schema mismatch | generated calls are syntactically plausible but invalid |
| synthetic-data bias | model inherits generator artifacts |
| unsafe memorization | private or sensitive data appears in outputs |

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
- Is the loss mask reported for chat templates and tool traces?
- Is evaluation separated by task slice rather than only one aggregate score?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[agents/core/agent-loop|Agent loop]]
