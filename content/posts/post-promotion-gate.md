---
title: Post Promotion Gate
tags:
  - posts
  - workflow
  - publishing
---

# Post Promotion Gate

Post promotion gate decides whether a cluster of wiki notes is ready to become a Korean reader-facing post. It prevents two failure modes: publishing a shallow post before the wiki is ready, or leaving a mature topic buried as disconnected notes.

$$
\text{post-ready}
=
\text{question}
\land
\text{wiki bundle}
\land
\text{evidence boundary}
\land
\text{public-safe}
\land
\text{next path}
$$

## Promotion Fields

| Field | Pass When |
| --- | --- |
| Reader question | the post answers one clear question |
| Primary axis | AI, Molecular Modeling, Math, Paper cluster, Project, Infra, or Agents is central |
| Topic map | broad posts have an explicit map contract rather than a loose link list |
| Wiki bundle | at least 3-5 reusable notes can be linked instead of redefined |
| Minimum formula | necessary equations are included with symbols, or linked to Math notes |
| Evidence boundary | paper, benchmark, split, metric, baseline, leakage, and uncertainty claims are scoped |
| Personal angle | the post adds reading order, interpretation, or practical judgment |
| Public boundary | private infrastructure, unpublished results, collaborator details, and internal tasks are absent |
| Next path | the reader has 3-7 useful links to continue |

## Wiki Bundle

| Bundle Type | Minimum Links |
| --- | --- |
| AI method post | architecture, learning method, objective, evaluation risk |
| Molecular Modeling post | entity/object, representation, preprocessing/split, evaluation risk |
| Math-heavy post | formula intake, explanation ladder, objective-metric link, evaluation math |
| Paper-cluster post | paper notes, claim/evidence table, benchmark contract, concept updates |
| Project post | project note, design decision, verification, related concepts |
| Infra post | public runbook, failure taxonomy, reproducibility or systems concept |
| Agent workflow post | agent workflow, verification loop, handoff or memory boundary |

## Decision Table

| Candidate State | Destination |
| --- | --- |
| one definition or equation | concept note |
| one paper claim | paper note |
| unverified source or unclear route | inbox or curation queue |
| reusable idea from a paper | concept update |
| several notes with one reader question | Korean post |
| implementation narrative | project note |
| public operational lesson | infra or public log |

## Draft Contract

Before drafting, fill this:

```yaml
reader_question: to verify
primary_axis: to verify
wiki_bundle:
  - to verify
minimum_formula: not applicable
evidence_boundary: to verify
paper_sources: not applicable
public_boundary: to verify
next_path:
  - to verify
status: draft
```

## Stop Conditions

- The post would mostly define one term.
- The post needs private context to make sense.
- The post has no reusable wiki links.
- The post is a broad map but does not pass [[concepts/topic-map-contract|Topic map contract]].
- The post depends on a paper claim whose metadata or evidence is still `to verify`.
- The post repeats long definitions that should live in Concepts, Papers, Math, or Infra.
- The post has no next reading path.

## Related

- [[posts/wiki-to-post-workflow|Wiki to post workflow]]
- [[posts/ai-bio-math-post-intake|AI-Molecular-Math post intake]]
- [[posts/synthesis-post-template|Synthesis post template]]
- [[posts/blog-writing-guide|Blog writing guide]]
- [[concepts/topic-map-contract|Topic map contract]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/ai-molecular-math-readiness-gate|AI-Molecular-Math readiness gate]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
