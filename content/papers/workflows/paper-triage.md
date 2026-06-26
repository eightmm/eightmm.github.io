---
title: Paper Triage
unlisted: true
aliases:
  - papers/paper-triage
tags:
  - papers
  - workflows
  - triage
---

# Paper Triage

Paper triage decides whether a candidate paper should be dropped, kept in the inbox, turned into a paper note, or used to update concepts. It is the filter between daily discovery and durable wiki growth.

A triage decision can be written as:

$$
d(p)
\in
\{\text{drop}, \text{inbox}, \text{paper note}, \text{concept update}, \text{synthesis}\}
$$

where $p$ is a candidate paper.

## Triage Criteria

- Source is public and metadata can be verified.
- Topic fits the site scope: AI, Bio, Infra, Agents, Papers, Projects, or Research.
- The paper introduces a reusable method, benchmark, failure mode, concept, or workflow.
- The claims can be tied to an evaluation protocol, benchmark, baseline, or reproducibility signal.
- The paper is not only interesting news; it changes a concept map or reading path.

## Decision Rules

- Drop: duplicate, unverifiable, out of scope, or only weakly relevant.
- Inbox: promising but metadata, claim, or relevance is not yet verified.
- Paper note: paper-specific contribution is worth tracking.
- Concept update: the idea is reusable and should strengthen an existing note.
- Synthesis: several verified papers support a broader map or Korean post.

## Paper Buckets

| Bucket | Use For |
| --- | --- |
| [Structure-based AI papers](/papers/sbdd) | docking, pose, scoring, screening, protein-ligand evaluation |
| [Protein modeling papers](/papers/protein-modeling) | protein representation, structure, antibody/protein interaction |
| [Generative model papers](/papers/generative-models) | diffusion, flow, molecule/protein generation, sampling objectives |
| [Learning method papers](/papers/learning-methods) | SSL, contrastive learning, JEPA, fine-tuning, preference/RL-style objectives |
| [Systems papers](/papers/systems) | training efficiency, inference, tooling, agents, reproducibility |
| Concept update only | math-heavy or method-definition paper where no paper-specific note is needed |

If a paper spans several buckets, choose the strongest claim as the paper location and use [[concepts/coverage-matrix|Coverage matrix]] for cross-links.

## Minimum Metadata

- Title.
- Public source link.
- Authors or source-provided author list.
- Date or version when available.
- Venue, preprint server, or repository status when available.
- Abstract-level topic and claimed contribution.

If metadata is missing, write `to verify` rather than inventing it.

## Checks

- What existing concept page would this update?
- Which paper bucket should hold the note?
- What claim would be extracted first?
- What benchmark, split, metric, or baseline does the paper rely on?
- What public artifacts are available: code, data, splits, configs, weights, logs, or predictions?
- Is there a likely leakage, data, reproducibility, or evaluation risk?
- Is it better as a brief inbox item than a full paper note?

## Related

- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/reading-status|Reading status]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
