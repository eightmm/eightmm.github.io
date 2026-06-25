---
title: Hallucination and Grounding
tags:
  - llm
  - grounding
  - evaluation
---

# Hallucination and Grounding

Hallucination is when a model produces unsupported or false content while presenting it as plausible. Grounding is the practice of tying outputs to evidence, tools, citations, retrieval, or verifiable state.

A grounded answer should satisfy:

$$
\text{claim}
\Rightarrow
\text{supporting evidence}
$$

For a set of generated claims $\mathcal{C}$ and evidence set $\mathcal{E}$, a simple audit question is:

$$
\forall c\in\mathcal{C},
\quad
\exists e\in\mathcal{E}
\text{ such that } e \text{ supports } c
$$

## Key Ideas

- Fluency is not evidence.
- Retrieval improves access to evidence but does not guarantee faithful use of it.
- Tool calls can ground answers in external state if tool results are checked.
- Missing evidence should produce uncertainty, abstention, or a narrower answer.
- In public notes, grounding means linking claims to concepts, papers, logs, or reproducible artifacts.
- A grounded answer can still be incomplete; grounding is about support, not exhaustiveness.

## Practical Checks

- Which claims need evidence?
- Is the evidence primary, current, and relevant?
- Does the answer distinguish sourced facts from inference?
- Are retrieved documents trusted as instructions or only as data?
- Is there a verification loop for high-risk outputs?
- Are unsupported claims removed, narrowed, or marked `to verify`?

## Related

- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[papers/claim-extraction|Claim extraction]]
