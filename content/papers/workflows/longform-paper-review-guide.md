---
title: Longform Paper Review Guide
unlisted: true
aliases:
  - papers/longform-paper-review-guide
tags:
  - papers
  - writing
  - workflow
---

# Longform Paper Review Guide

Use this guide before writing a public paper review that should read like a blog post rather than a compact wiki note. The target is a beginner-friendly review that a reader can finish in about 20-30 minutes while still leaving enough structure for later wiki reuse.

This guide is especially appropriate for foundational papers such as architecture, learning-method, generative-model, evaluation, or computational biology papers.

## Review Goal

A longform paper review should answer one reader question:

$$
\text{paper}
\rightarrow
\text{problem}
\rightarrow
\text{idea}
\rightarrow
\text{method}
\rightarrow
\text{evidence}
\rightarrow
\text{limits}
\rightarrow
\text{why it matters}
$$

It should not be a section-by-section translation of the paper. It should explain why the paper was needed, what changed, how the method works, what evidence supports it, and which concepts the reader should learn next.

## Length and Reading Budget

| Target | Guideline |
| --- | --- |
| Reading time | 20-30 minutes |
| Main sections | 7-10 sections |
| Math depth | enough to understand the central claim |
| Tables | use for comparison, routing, evidence, limitations |
| Figures | use self-made diagrams, Mermaid diagrams, or concise schematic descriptions |
| External claims | cite or mark `to verify`; do not invent metadata, metrics, or results |

If the review becomes longer than this, split reusable definitions into [[concepts/index|Concepts]], formulas into [[math/index|Math]], and evidence details into [[papers/analysis/index|Paper analysis]].

## Required Structure

Use this shape unless the paper strongly suggests a better one.

| Section | Purpose | Typical Links |
| --- | --- | --- |
| One-line summary | state the paper's core contribution in plain language | paper bucket, main concept |
| Why this paper mattered | explain the problem before the method | [AI](/ai), [Math](/math), [Computational Biology](/molecular-modeling) when relevant |
| Background before the paper | define the baseline or older paradigm | architecture, learning, evaluation, modality notes |
| Paper's main idea | name the key move and what it replaces | concept notes and formulas |
| Method walkthrough | explain components in dependency order | equations, diagrams, route tables |
| Experiments and evidence | connect claims to tables, benchmarks, baselines, and ablations | [Evidence table](/papers/analysis/evidence-table), [Benchmark card](/papers/analysis/benchmark-card) |
| What the paper does not prove | limits, assumptions, missing comparisons, leakage or scaling caveats | [Limitation taxonomy](/papers/analysis/limitation-taxonomy) |
| Concept map | list the wiki pages the reader should open next | AI, Math, Computational Biology, Infra, Agents |
| Takeaways | explain what should remain in memory after reading | related papers or project ideas |
| Open questions | mark uncertainty, reproduction targets, or follow-up reading | reproducibility and research-method notes |

## Beginner-Friendly Writing Rules

- Start with the problem, not the model name.
- Define every important object before using equations.
- Explain what each equation computes and why it appears.
- Prefer the paper's causal story over a chronological summary.
- Use short paragraphs and descriptive section titles.
- Put comparisons into tables rather than long prose.
- Use diagrams when several tensors, modules, or evidence paths interact.
- Do not paste copyrighted figures from the paper. Recreate minimal schematic diagrams or describe the figure's role.
- Do not copy long passages from the paper. Paraphrase and link the public source.
- Mark unknown metadata, artifact status, and uncertain claims as `to verify`.

## Formula Depth

Use [[math/formula-explanation-ladder|Formula explanation ladder]] before writing formulas.

| Case | Depth |
| --- | --- |
| Formula is background only | define the intuition and link to Math |
| Formula is the paper's main contribution | include the full canonical equation and symbol definitions |
| Formula explains a model block | include input, intermediate variables, output, and shape or dimension when useful |
| Formula explains evaluation | include metric definition, denominator, split, and what higher/lower means |

For architecture papers, avoid shorthand. For attention, define:

$$
Q = XW_Q,\quad
K = XW_K,\quad
V = XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

Then explain what $Q$, $K$, $V$, $d_k$, softmax, masking, heads, and output projection do. If a causal or padding mask is used, include the mask term:

$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

## Visual and Table Policy

Use visuals to reduce cognitive load, not to decorate.

| Need | Preferred Form |
| --- | --- |
| Compare old vs new method | two-column table |
| Explain a pipeline | Mermaid flowchart |
| Explain architecture blocks | compact schematic or ordered block table |
| Explain experiment claims | claim-evidence table |
| Explain benchmark limits | benchmark card |
| Explain formulas | equation plus symbol table |

For longform reviews, at least one of these should usually appear:

- problem-to-method table;
- method component table;
- formula symbol table;
- evidence table;
- limitation table;
- concept map.

## Wiki Link Policy

Every longform paper review should include links that help the reader leave the post when they need prerequisites.

| Link Type | Use |
| --- | --- |
| AI | architecture, learning method, generative model, evaluation, systems |
| Math | probability, linear algebra, optimization, information theory, geometry, metrics |
| Computational Biology | protein, molecule, ligand, structure, interaction, docking, data/evaluation |
| Concepts | reusable definitions and formulas |
| Papers | related papers and comparisons |
| Infra | compute, training, serving, reproducibility, systems claims |
| Agents | agent workflow or tool-use papers |

Do not repeat full prerequisite tutorials inside the review. Link the canonical note, then explain only the minimum needed for this paper.

## Evidence Discipline

Use the paper's evidence only within the scope it supports.

| Evidence Item | Must State |
| --- | --- |
| Dataset or benchmark | task, split, metric, allowed information |
| Result table | metric direction, baseline, selection rule, uncertainty when available |
| Ablation | what component was isolated and what remained controlled |
| Scaling result | parameter count, data, compute, sequence length, hardware or implementation boundary |
| Artifact | public code, data, configs, weights, logs, predictions, environment |

When the paper does not provide enough information, write `to verify` rather than guessing.

## Longform Review Skeleton

```markdown
---
title: Paper Title
aliases:
  - papers/short-slug
tags:
  - papers
  - ai
---

# Paper Title

> One-line summary.

## Metadata

| Field | Value |
| --- | --- |
| Paper | to verify |
| Authors | to verify |
| Venue / year | to verify |
| Link | to verify |
| Status | reading |

## Why This Paper Mattered

Problem before the paper.

## Background

What a beginner must know first.

## Main Idea

The core move.

## Method Walkthrough

Component-by-component explanation.

## Core Formula

Equation, symbol table, and intuition.

## Experiments and Evidence

| Claim | Evidence | Caveat |
| --- | --- | --- |
| to verify | to verify | to verify |

## Limitations

What the paper does not prove.

## Concept Map

- [[ai/index|AI]]
- [[math/index|Math]]
- [[papers/index|Papers]]

## Takeaways

What to remember.

## Open Questions

What remains unresolved.
```

## Pre-Publish Checklist

- Metadata is verified or marked `to verify`.
- The paper's problem is clear before method details.
- The main contribution is stated in one sentence.
- The method section is ordered by dependency, not by paper section order.
- Central formulas include symbol definitions.
- At least one table or diagram improves readability.
- Claims are tied to evidence, baseline, metric, split, or ablation.
- Limitations and unsupported claims are explicit.
- AI, Math, Computational Biology, Infra, or Agent links are added where useful.
- No private project details, unpublished results, private paths, server details, or credentials appear.
- The note can be read as a public blog-style explanation in 20-30 minutes.

## Related

- [[papers/index|Papers]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/paper-to-wiki-extraction|Paper to wiki extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[concepts/coverage-matrix|Coverage matrix]]
