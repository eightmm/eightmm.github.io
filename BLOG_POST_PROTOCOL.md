# BLOG_POST_PROTOCOL.md

A readable operating protocol for writing paper-review posts on `eightmm.github.io`.

This document is meant to be easy to follow during real work, not just correct in theory.

If asked to write a paper post, default to this protocol unless the user asks for something different.

---

## 1. Quick Defaults

### Always

- Write internal protocol and planning in English.
- Write the final blog post in Korean by default.
- Prefer a deep explanatory review, not a short summary.
- Include at least one representative image when available.
- Prefer rendering stability over fancy formatting.
- Treat architecture and pipeline explanation as first-class content, not optional garnish.

### Default

- Long-form technical review
- Clear section structure
- 1 to 3 useful figures
- 3 to 8 important equations if the paper is math-heavy
- Bullets over fragile tables when possible
- `mermaid: false` unless clearly needed
- At least one architecture-oriented code or pseudocode sketch

### Avoid

- abstract paraphrase
- hype tone
- fragile Markdown tables
- dense inline math
- future-dated posts
- empty front matter image when a good figure exists
- posts that explain results but barely explain the model

---

## 2. What Good Looks Like

A good post should let a reader do at least three of these:

- explain the paper to someone else
- restate the main equations
- reconstruct the rough implementation flow
- say what changed versus prior methods
- point out the paper’s actual limitations
- describe the main architecture blocks and why they are arranged that way

The target is:

> a strong technical colleague walking the reader through the paper, with judgment, structure, and enough detail to be useful.

---

## 3. Workflow

Follow the work in this order.

### Step 1: Before Writing

Collect the minimum required information:

- paper title
- paper URL
- authors
- affiliations
- venue or publication status
- publication date
- one key overview figure
- one key result figure or table

Collect these too when useful:

- appendix or supplementary
- project page
- code repo
- ablations
- failure cases
- theorem / proposition / lemma
- runtime, parameter count, dataset size
- architecture figure
- training or inference pipeline figure

Verify before drafting:

- important numbers are correct
- notation is copied correctly
- metric definitions are not mixed up
- claims are separated from evidence
- comparisons are fair
- the architecture story is actually understood, not just namedropped

### Step 2: While Drafting

Default structure:

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

Writing goals by section:

- `Hook`: why the paper matters
- `Problem`: 2 to 4 real bottlenecks
- `Key Idea`: the compressed contribution
- `How It Works`: the actual core, with equations, architecture, and implementation logic
- `Results`: interpretation, not number dumping
- `Discussion`: what it means and where it fits
- `Limitations`: concrete reasons to be cautious
- `TL;DR`: fast scan summary
- `Paper Info`: links and metadata

### Step 3: Before Push

Run a quick preflight:

- publish `date` is not in the future
- front matter is complete
- representative image exists
- math blocks are separated cleanly
- no unnecessary Mermaid
- no fragile wide tables unless truly needed
- links resolve
- architecture section is not too thin relative to the rest of the post
- post reads cleanly once for content and once for rendering risk

### Step 4: After Deploy

Check:

- GitHub Pages or deploy workflow succeeded
- the post is actually visible
- image loads
- equations render
- code blocks look fine
- spacing is clean
- mobile overflow is not obvious

If the post is missing, check the publish date first.

---

## 4. Architecture and Pipeline Rules

This section is mandatory in spirit even if the exact heading names vary.

### What must be explained

For architecture-heavy papers, the post should clearly explain:

- the main representations used by the model
- the major blocks or stages
- the order in which those blocks run
- what information each block consumes and produces
- why the architecture is shaped that way
- what changed relative to the closest prior baseline

### Minimum architecture coverage

A good `How It Works` section should usually include most of these:

- one overview of the full pipeline
- one representation subsection
- one core architecture subsection
- one algorithm / training / inference subsection
- one short code or pseudocode sketch
- one paragraph on why the design should work

### Preferred explanation pattern

When describing a model block, prefer this order:

1. what goes in
2. what the block computes
3. what comes out
4. why that matters
5. how it differs from prior work

### Good default questions

If stuck, answer these explicitly:

- What is the trunk?
- What is the generator / decoder / structure module?
- Where does geometric reasoning happen?
- Where does pair information enter?
- What is iterative and what is one-shot?
- What used to be external optimization, and is now inside the model?

### Architecture-first warning signs

If the draft has any of these, it is probably too shallow:

- block names are listed but not explained
- architecture figure exists but is not unpacked
- the post says a model is diffusion / transformer / graph-based without explaining the actual flow
- results are detailed but the model internals are vague
- the reader still cannot sketch the pipeline after reading

---

## 5. Safe Formatting Rules

### Math

Use inline math only for short, safe expressions.

Move math into display blocks when it has:

- many subscripts or superscripts
- `\hat`, `\tilde`, `\mathcal`, `\nabla`, `\otimes`
- long conditional notation
- enough complexity to break reading flow

If a paragraph gets equation-heavy, split it.

### Tables

Use Markdown tables only when the structure really matters.

Prefer bullets when:

- the comparison is short
- the table would be wide
- the theme may render it poorly
- the same information fits naturally as nested bullets

Rule:

> If a comparison fits in bullets, do not force a table.

### Mermaid

Do not enable Mermaid by default.

Use Mermaid only when:

- it adds real explanatory value
- the diagram is simple
- the same idea cannot be explained more robustly with bullets plus an image

Rule:

> If rendering stability and prettiness conflict, choose rendering stability.

### Images

Prefer local files under:

- `assets/img/posts/<slug>/`

Use images with clear roles:

- overview figure
- method figure
- results figure
- ablation or failure case figure

Every image should have meaningful alt text.

Prefer a local copy over a hotlinked external asset when practical.

---

## 6. Front Matter Defaults

Use this as the safe default:

```yaml
---
title: "Paper title"
date: YYYY-MM-DD HH:MM:SS +0900
description: "Compressed summary of contribution, mechanism, and main result"
categories: [AI, Subcategory]
tags: [tag1, tag2, tag3, tag4]
math: true
mermaid: false
image:
  path: /assets/img/posts/<slug>/fig1_overview.png
  alt: "Representative figure description"
---
```

### Front matter rules

- `description` should be concise and informative
- do not paste the abstract
- use `mermaid: true` only when necessary
- include `image` whenever a meaningful representative figure exists
- use a conservative publish time if posting immediately

---

## 7. Content Quality Rules

### Claim hygiene

Do not write stronger than the evidence.

If using words like these:

- SOTA
- state-of-the-art
- significant
- strong
- substantially better

then back them with:

- numbers
- comparison conditions
- scope limitations when relevant

Separate:

- what the paper claims
- what the evidence actually supports

If a result holds only in a narrow setup, say so directly.

### Depth standard

Default to a deep explanatory review, not a summary note.

In practice this means:

- `How It Works` should usually be the largest section
- key equations should be explained, not merely copied
- architecture should be described as a flow:
  - input
  - representation
  - update
  - output
- results should include interpretation
- limitations should be explicit
- architecture-heavy papers should devote meaningful space to block-level explanation

---

## 8. File Rules

### Post location

- `_posts/YYYY-MM-DD-slug.md`

### Image location

- `assets/img/posts/<slug>/`

### Slug rules

- use kebab-case
- keep only the most meaningful terms
- shorten when necessary

### Recommended figure names

- `fig1_overview.png`
- `fig2_method.png`
- `fig3_results.png`
- `fig4_ablation.png`
- `fig5_failure_cases.png`

---

## 9. Paper Info Rules

Prefer bullets over tables when rendering stability is a concern.

Include when available:

- title
- authors
- affiliations
- venue
- publication date
- paper URL
- project URL
- code URL

---

## 10. Quick Checklists

### Drafting checklist

- Did I explain why this paper matters?
- Did I identify real bottlenecks, not just say prior work was weaker?
- Did I explain how the method works in implementation terms?
- Did I explain the architecture blocks, not just mention them?
- Did I make the pipeline order clear?
- Did I interpret the results?
- Did I state real limitations?

### Rendering checklist

- Is Mermaid really needed?
- Is any table too wide?
- Is any inline math too dense?
- Is the front matter image present?
- Are spacing and blocks visually clean?

### Deployment checklist

- Is the post date safe?
- Did deploy succeed?
- Is the page visible?
- Do image, math, and code blocks render correctly?

---

## 11. If Asked Repeatedly, Keep Following This

If the user asks again to turn a paper into a blog post, keep using these defaults unless told otherwise:

- internal instructions in English
- final post in Korean
- deep review structure
- rendering-safe formatting
- representative image included
- architecture and pipeline explanation included
- post-deploy verification

This protocol is meant to be the stable default path.

For concrete examples, see `BLOG_POST_EXAMPLES.md`.

Useful scripts:

- `python3 scripts/new_blog_post.py ...` to create a scaffold
- `python3 scripts/validate_blog_post.py <post.md>` for structural validation
- `python3 scripts/post_preflight.py <post.md>` for protocol-aligned preflight checks
- `scripts/check_blog_post.sh <post.md>` to run the common checks together
- `python3 scripts/validate_blog_rendered.py --url https://eightmm.github.io --posts <slug>` for rendered-page validation
