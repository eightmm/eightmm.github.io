---
title: Evidence-Grounded Generation
tags:
  - llm
  - grounding
  - evaluation
---

# Evidence-Grounded Generation

Evidence-grounded generation is the practice of requiring generated claims to be supported by explicit evidence. It is stricter than retrieval-augmented generation: retrieval supplies candidate context, while grounding checks whether the final output is actually supported.

A generated answer can be viewed as claims:

$$
y \rightarrow \mathcal{C}(y)=\{c_1,\ldots,c_n\}
$$

Grounding asks whether each claim has support:

$$
\forall c_i\in\mathcal{C}(y),
\quad
\exists e_j\in\mathcal{E}
\text{ such that }
e_j \models c_i
$$

where $\mathcal{E}$ is the evidence set and $\models$ means "supports."

## Evidence Types

- Retrieved document passage.
- Tool result.
- Source paper metadata.
- Local file or build output.
- Reproducible run record.
- Human-reviewed note.

## Output Contract

A grounded answer should separate:

- Supported facts.
- Inferences from supported facts.
- Unverified claims marked `to verify`.
- Missing evidence.
- Next verification step.

## Failure Modes

- Retrieved evidence is relevant but the answer adds unsupported claims.
- Evidence is present but outdated, low quality, or not primary.
- The model treats untrusted retrieved text as instructions.
- Citations point to sources that do not support the sentence.
- A fluent synthesis hides uncertainty or missing data.

## Checks

- Which claims require evidence?
- Is the evidence primary enough for the claim?
- Are citations or wikilinks attached to the exact claim they support?
- Are unsupported claims narrowed or marked `to verify`?
- Is there a verification loop for high-impact content?

## Related

- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[papers/evidence-table|Evidence table]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
