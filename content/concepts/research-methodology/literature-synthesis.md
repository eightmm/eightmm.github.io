---
title: Literature Synthesis
tags:
  - research
  - papers
  - methodology
---

# Literature Synthesis

Literature synthesis turns multiple paper notes into a reusable research map. It is different from summarizing one paper: the goal is to compare claims, evidence, limitations, and open questions across related work.

A synthesis note can be viewed as:

$$
\mathcal{S}
=
F(\{p_i\}_{i=1}^{n}, \mathcal{Q})
$$

where $p_i$ are paper notes and $\mathcal{Q}$ is the research question being clarified.

## Synthesis Axes

- Task: what problem is being solved?
- Input and output: what entities, modalities, or structures are modeled?
- Architecture: what inductive bias is used?
- Objective: what training signal is optimized?
- Data: what benchmark, split, and preprocessing are used?
- Evidence: which claims are supported by which experiments?
- Limitation: what remains uncertain or weak?
- Reusable concept: what should be promoted into a concept note?

## Process

1. Start from a concrete [[concepts/research-methodology/research-question|research question]].
2. Select a small set of papers with comparable tasks.
3. Extract claims with [[papers/analysis/claim-extraction|Claim extraction]].
4. Map evidence with [[papers/analysis/evidence-table|Evidence table]].
5. Compare methods with [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]].
6. Convert stable ideas into [[concepts/index|Concepts]].
7. Write a Korean post only when the synthesis has a clear reader path.

## Checks

- Are papers compared on the same task and metric?
- Are missing values marked rather than guessed?
- Does the synthesis narrow broad claims?
- Does it expose a real research gap?
- Are copied abstracts avoided?

## Related

- [[papers/index|Papers]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[concepts/research-methodology/research-question|Research question]]
- [[posts/workflows/topic-roadmap|글감 로드맵]]
