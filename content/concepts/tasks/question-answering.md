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

## Checks

- What evidence is allowed?
- Is the answer directly supported by the context?
- Are citations or source spans required?
- Does the model abstain when evidence is missing?
- Is evaluation measuring exact answer, reasoning, retrieval quality, or citation quality?

## Related

- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[agents/verification-loop|Verification loop]]
