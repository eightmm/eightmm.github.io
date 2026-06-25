---
title: Structured Output
tags:
  - llm
  - structured-output
  - workflows
---

# Structured Output

Structured output constrains a language model response to a schema such as JSON, YAML, a table, a function call, or a domain-specific format. It makes outputs easier to parse, validate, test, and feed into tools.

If $\mathcal{Y}_{\mathrm{valid}}$ is the set of valid outputs, constrained generation aims for:

$$
\hat{y}
\in
\mathcal{Y}_{\mathrm{valid}}
$$

One way to view decoding is:

$$
\hat{y}
=
\arg\max_{y\in\mathcal{Y}_{\mathrm{valid}}}
p_\theta(y\mid x)
$$

where invalid outputs are excluded or rejected.

## Key Ideas

- A schema is a contract between the model and downstream code.
- Valid syntax is not the same as correct semantics.
- Structured output reduces parsing ambiguity but does not remove the need for validation.
- Tool arguments should be checked before execution.
- For public wiki workflows, structured output can turn paper reading, claim extraction, and link suggestions into reviewable artifacts.

## Practical Checks

- What schema must the output satisfy?
- Are required fields, enums, ranges, and nested objects explicit?
- Are invalid outputs rejected, repaired, or retried?
- Does validation check semantics, not only syntax?
- Is untrusted model output isolated before it affects tools or files?
- Are evidence fields required when the output contains factual claims?

## Related

- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/tool-calling|Tool calling]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
