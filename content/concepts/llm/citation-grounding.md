---
title: Citation Grounding
tags:
  - llm
  - grounding
  - evaluation
---

# Citation Grounding

Citation grounding checks whether cited evidence actually supports the generated claim. A citation is not enough by itself; the linked source must entail or justify the sentence it is attached to.

Represent an answer as claims and citations:

$$
y
\rightarrow
\{(c_i, E_i)\}_{i=1}^{n}
$$

where $c_i$ is a claim and $E_i$ is the cited evidence set. Grounding requires:

$$
E_i
\models
c_i
$$

for each claim that needs support.

## Citation Roles

Citations can support different roles:

- source of a factual statement;
- evidence for a measurement or metric;
- provenance for a quote or dataset;
- pointer to an implementation artifact;
- background reference for a definition.

The role should match the strength of the claim. A background source should not be used as proof of a new result.

## Failure Modes

- Citation points to a relevant source, but not the exact claim.
- Citation supports a weaker claim than the answer states.
- Citation is outdated for a version-sensitive fact.
- Citation is secondary when a primary source is needed.
- Citation appears after a paragraph but supports only one sentence.
- Generated text combines multiple sources into an unsupported inference.

## Checks

- Which sentence or claim does each citation support?
- Is the source primary enough for the claim?
- Does the evidence support the exact scope, date, and wording?
- Are inferences marked separately from cited facts?
- Are missing or uncertain facts marked `to verify`?

## Related

- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/tasks/question-answering|Question answering]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/verification-loop|Verification loop]]
