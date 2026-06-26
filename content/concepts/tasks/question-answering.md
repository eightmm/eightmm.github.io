---
title: Question Answering
tags:
  - tasks
  - language-models
  - retrieval
---

# Question Answering

Question answering returns an answer conditioned on a question and optional evidence. The key distinction is whether the answer is extracted, generated, retrieved, or computed.

A retrieval-augmented setup can be written as:

$$
d_{1:k} = \operatorname{Retrieve}(q, \mathcal{D})
$$

$$
\hat{a} = \operatorname{Generate}(q, d_{1:k})
$$

where $q$ is the question, $\mathcal{D}$ is the evidence corpus, and $d_{1:k}$ are retrieved documents.

## Types

- Closed-book QA: answer from model parameters.
- Open-book QA: answer from retrieved evidence.
- Extractive QA: select a span from context.
- Generative QA: synthesize an answer.
- Multimodal QA: answer over image, video, table, molecule, or structure inputs.

## Evidence Contract

For public and research workflows, QA should define what evidence is allowed:

$$
\hat{a}
=
f_\theta(q,E),
\qquad
E
\subseteq
\mathcal{D}_{\mathrm{allowed}}
$$

where $E$ may include retrieved documents, tool outputs, source files, figures, tables, or user-provided context. If the answer cannot be supported by $E$, the correct behavior may be abstention rather than generation.

## Answer Types

The output space may be:

$$
\mathcal{Y}
\in
\{\text{span},
\text{short answer},
\text{long answer},
\text{yes/no},
\text{list},
\text{calculation},
\text{citation set},
\text{tool result}\}
$$

Each answer type needs a different metric and validity check. Exact match is reasonable for short factual answers, but weak for synthesis, calculations, or evidence-grounded explanations.

## Retrieval and Answer Coupling

A QA failure can happen in retrieval or generation:

$$
\Pr(\text{correct answer})
\le
\Pr(\text{evidence retrieved})
$$

If necessary evidence is absent from retrieved context, generation quality cannot fix the task. Evaluate retrieval recall, answer correctness, and citation support separately.

## Checks

- What evidence is allowed?
- Is the answer directly supported by the context?
- Are citations or source spans required?
- Does the model abstain when evidence is missing?
- Is evaluation measuring exact answer, reasoning, retrieval quality, or citation quality?
- Is unsupported answer generation counted as failure even when fluent?
- Is the question answerable from the allowed evidence set?
- Are retrieval misses separated from reasoning or synthesis errors?
- Are citations checked for support, not only presence?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/citation-grounding|Citation grounding]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[agents/verification/verification-loop|Verification loop]]
