# BLOG_POST_PROTOCOL.md

This document defines the standard protocol for long-form paper review posts on `eightmm.github.io`.

The goal is not a shallow summary. The default target is a **deep explanatory technical review** that helps a reader understand:

- why the paper matters,
- what problem it solves,
- what the key modeling choices are,
- how the method actually works,
- what the results mean,
- and where the limitations are.

In one sentence:

> A reader who has not read the paper should still be able to follow the core idea, the main equations, the implementation logic, and the real limitations from this post alone.

---

## 0. Core Principles

Every paper review should aim for these five things:

1. **Context**: why this problem matters now
2. **Claim**: what the authors actually changed
3. **Mathematical structure**: what space, objective, or probabilistic view the paper uses
4. **Implementation intuition**: what each module does and how the pieces interact
5. **Evaluation and limits**: how convincing the evidence is, and where caution is needed

Avoid:

- abstract paraphrase
- promo tone
- equation dumping without explanation
- shallow praise like “strong”, “good”, “impressive” without analysis

Prefer:

- intuition + equations + structure + criticism in the same post

---

## 1. Output Language Policy

- **Internal protocol and prompting should be written in English.**
- **The final blog post should normally be written in Korean**, unless explicitly requested otherwise.
- Technical notation, code, and paper titles can remain in English when that improves clarity.

Why:

- English internal instructions tend to produce more stable structure and cleaner technical reasoning.
- Korean final output keeps the actual blog aligned with the intended readership and voice.

---

## 2. Standard Post Style

The default style is **deep, seminar-note-like, long-form technical review**.

This means:

- `How It Works` should be substantial
- key equations should be connected to the paper’s notation whenever possible
- important propositions, theorems, or lemmas should be explained, not just mentioned
- forward process, objective, and inference should be separated clearly when relevant
- architecture should be described as a flow:
  - input
  - representation
  - update
  - output
- result sections should interpret why the numbers matter, not just restate them

A good post should allow the reader to do at least three of these:

- explain the paper to a colleague
- rewrite the main equations on a whiteboard
- roughly reconstruct the implementation flow
- state the difference from strong baselines
- point out the paper’s limitations concretely

---

## 3. Length and Density Guidelines

### Length

- Default target: **long post**
- Prefer seminar-note level depth over a short review note

### Density

- `How It Works`: ideally **40 to 55 percent** of the post
- Equations: usually **3 to 8 important equations** when the paper is math-heavy
- Code blocks: **at least 1**, ideally **2 or more** when useful
- Figures: **at least 1 representative figure**, preferably more if they genuinely help

### Priority order

1. correctness
2. explanation of design choices
3. implementation intuition
4. reading flow
5. decorative style

---

## 4. Rendering Safety Rules

This is important. Prefer formats that survive Jekyll / Markdown / KaTeX / theme rendering reliably.

### Math

- Use inline math only for **short and safe expressions**.
- Move anything complex into display math blocks.
- Avoid overly dense inline expressions with:
  - many subscripts or superscripts
  - `\hat`, `\tilde`, `\mathcal`, `\nabla`, `\otimes`
  - conditional probability notation inside long prose
- If one paragraph contains too many equations, split it.

### Tables

- Use Markdown tables only when the structure truly matters.
- If a table is simple, prefer bullet lists instead.
- If rendering risk is high, rewrite comparison tables as nested bullets.

### Mermaid

- Do **not** enable Mermaid by default.
- Use Mermaid only when it clearly adds value and the diagram is simple.
- If a concept can be explained cleanly with bullets plus one image, prefer that.

### Images

- Prefer local images under `assets/img/posts/<slug>/`
- Always add a front matter image when possible
- Include meaningful alt text
- If an external figure is used as inspiration, download and store a local copy when licensing and practical constraints allow

### Safe structure

- Keep enough blank lines between headings, lists, code blocks, and equations
- Avoid mixing long bullet lists and complex display math without spacing
- Avoid very wide tables if the theme may overflow on mobile

---

## 5. What to Collect Before Writing

Required:

1. paper title
2. paper link
3. authors and affiliations
4. publication date and venue
5. one key overview figure
6. one key result figure or table

Strongly recommended:

- appendix or supplementary material
- project page
- code repository
- ablation table
- failure case figure
- theorem / proposition / lemma
- train-test split details
- runtime, parameter count, and dataset size

Always verify:

- important numbers are correct
- notation is copied correctly
- baseline comparison is fair
- metric definitions are not mixed up
- paper claims are distinguished from evidence

---

## 6. File and Asset Rules

### Locations

- Post: `_posts/YYYY-MM-DD-slug.md`
- Images: `assets/img/posts/<slug>/`

### Slug

- use kebab-case
- shorten when necessary
- keep the most meaningful terms

### Recommended image names

- `fig1_overview.png`
- `fig2_method.png`
- `fig3_results.png`
- `fig4_ablation.png`
- `fig5_failure_cases.png`

### Figure policy

- Do not impose an artificial hard limit on figure count
- Use multiple figures when they materially improve understanding
- Prefer figures with clear roles:
  - overall overview
  - method figure
  - notation-supporting figure
  - results / ablations / failure cases
- Avoid redundant figures that repeat the same information

---

## 7. Front Matter Rules

Base template:

```yaml
---
title: "Paper title"
date: YYYY-MM-DD HH:MM:SS +0900
description: "Compressed summary of contribution, mechanism, and key result"
categories: [AI, Subcategory]
tags: [tag1, tag2, tag3, tag4]
math: true
mermaid: false
image:
  path: /assets/img/posts/<slug>/fig1_overview.png
  alt: "Representative figure description"
---
```

Notes:

- Do not paste the abstract into `description`
- `description` should normally be concise and informative
- Enable `mermaid: true` only when necessary
- Prefer a real representative image instead of leaving `image` empty

---

## 8. Standard Document Structure

Use this structure by default:

```md
## Hook
## Problem
## Key Idea
## How It Works
## Results
## Discussion
## Limitations
## Conclusion
## TL;DR
## Paper Info
```

Guidelines:

- `Hook`: short, sharp, motivating
- `Problem`: explain the structural bottlenecks
- `Key Idea`: compressed statement of the paper’s contribution
- `How It Works`: the main body, detailed and careful
- `Results`: interpretation, not number dumping
- `Discussion`: meaning, positioning, comparison
- `Limitations`: separate section, explicitly critical

---

## 9. Recommended Writing Pattern by Section

### Hook

Explain:

- current state of the field
- limitation of existing methods
- the paper’s central claim
- what this post will unpack

### Problem

Break the problem into **2 to 4 bottlenecks**.
Focus on structural causes, for example:

- representation bottleneck
- complexity bottleneck
- inductive bias mismatch
- objective mismatch
- train-test leakage
- physical validity constraints

### Key Idea

State the paper’s contribution in the most compressed useful form.
If possible, include a direct before-versus-after comparison with prior paradigms.

### How It Works

This is the core section.
Describe:

- representation
- objective
- architecture
- optimization or training recipe
- inference or sampling procedure
- why the design should work

When useful, include:

- one overview image
- one pseudocode block
- one minimal implementation sketch

### Results

Do not just list numbers.
Explain:

- what is actually improved
- whether the comparison is meaningful
- why a specific metric matters
- whether the result changes the practical picture

### Discussion

Say what the paper means.
Possible angles:

- how it reframes the problem
- what assumptions it breaks
- how it compares philosophically to neighboring methods
- where it may matter in practice

### Limitations

Be concrete.
Possible categories:

- weak empirical scope
- dependence on a specific dataset or benchmark
- expensive training recipe
- unclear generalization
- theory not fully closed
- ecosystem or tooling gap

### Conclusion

Summarize the paper’s real contribution in a few strong paragraphs.
Avoid hype.

### TL;DR

Use concise bullet points.
These should be easy to scan and preserve the core message.

### Paper Info

Prefer bullets over tables when rendering stability is a concern.
Include:

- title
- authors
- affiliations
- venue
- publication date
- paper URL
- project URL
- code URL if relevant

---

## 10. Post-Write Checklist

Before committing:

- confirm the publish `date` is not in the future relative to deployment time
- make sure front matter includes a representative image when available
- check that math blocks are separated cleanly
- replace fragile tables with bullets if unnecessary
- disable Mermaid unless it clearly helps
- confirm links, names, and metrics
- skim the post once for flow and once for rendering risk

### 10.1 Publish-Time Safety

- Never publish a post with a future `date` unless intentional scheduling is explicitly desired.
- Compare the post timestamp against local deployment time, not just the paper date.
- If publishing immediately, prefer a conservative earlier timestamp on the same day.
- If there is any doubt, choose a timestamp safely in the past.

### 10.2 Rendering Preflight Checklist

Before pushing, quickly scan for rendering risks:

- Is Mermaid truly necessary?
- Is there any wide Markdown table that should become bullets?
- Is any inline math too complex and better moved into display math?
- Is there any block likely to overflow on mobile?
- Are heading, list, code block, and equation spacings visually clean?
- Is any comparison being forced into a table when bullets would be safer?

Rules of thumb:

- If rendering stability and prettiness conflict, choose rendering stability.
- If a comparison fits in bullets, do not force a table.

### 10.3 Image Policy

- A representative image should be included whenever a meaningful one is available.
- Prefer a local image under `assets/img/posts/<slug>/` rather than a hotlinked external asset.
- If the project page has a useful overview figure, consider storing a local copy when practical.
- Every image should have meaningful alt text.
- Prefer figures that explain the method or summarize the result, not decorative filler.

### 10.4 Link Policy

Every paper post should include, at minimum when available:

- paper URL
- project page URL
- code repository URL

Recommended:

- include the main paper link in `Paper Info`
- include the project page in `Paper Info`
- if useful, include one contextual link near the first relevant mention in the body
- verify that links resolve correctly before pushing

### 10.5 Claim Hygiene

- Do not copy the abstract’s strongest claims verbatim unless clearly attributed.
- Words like `SOTA`, `state-of-the-art`, `significant`, `strong`, or `substantially better` should be backed by numbers and comparison conditions.
- Separate what the paper claims from what the evidence actually shows.
- If a result holds only in a narrow setting, say that directly.
- Avoid overstating practical impact when the paper only establishes benchmark gains.

### 10.6 Post-Deploy Verification

After pushing:

- check GitHub Pages or deployment workflow status
- confirm the post is actually visible
- open the deployed page and inspect:
  - representative image
  - equations
  - code blocks
  - bullets and spacing
  - mobile overflow risk
- if the page is hidden unexpectedly, check publish date first

---

## 11. Final Reminder

The blog should feel like a strong technical colleague walking the reader through the paper, not like a translated abstract or a hype post.

Depth matters.
Clarity matters.
Judgment matters.
Rendering stability also matters.
