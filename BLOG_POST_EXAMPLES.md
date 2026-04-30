# BLOG_POST_EXAMPLES.md

Concrete examples that support `BLOG_POST_PROTOCOL.md`.

Use this file for patterns, not as a rigid template.

---

## 1. Safe Front Matter Example

```yaml
---
title: "Generative Modeling via Drifting: one-step 생성은 inference가 아니라 training에서 만들어진다"
date: 2026-04-30 11:40:00 +0900
description: "Drifting Models는 inference-time iteration을 training-time pushforward evolution으로 옮기고, one-step generation에서 strong ImageNet FID를 보인다."
categories: [AI, Generative Models]
tags: [generative-modeling, diffusion, flow-matching, one-step-generation, imagenet]
math: true
mermaid: false
image:
  path: /assets/img/posts/generative-modeling-via-drifting/fig1_overview.png
  alt: "Drifting Models overview and generated ImageNet samples"
---
```

Notes:

- `mermaid: false` is the safer default.
- Use a publish time that is not in the future.
- Include a real image when possible.

---

## 2. Good Description Examples

### Better

- `Mamba-3는 더 표현력 있는 discretization, complex-valued state update, MIMO SSM으로 linear sequence model의 성능-효율 frontier를 다시 민다.`
- `Drifting Models는 inference-time iteration 대신 training-time pushforward evolution을 사용해 one-step generation을 학습하고, ImageNet 256x256에서 strong FID를 기록한다.`

### Worse

- `이 논문은 새로운 생성 모델을 제안한다.`
- `아주 강력한 방법을 보여준다.`
- abstract pasted into description

Why the better versions work:

- they say what changed
- they say how it changed
- they say why it matters

---

## 3. Table vs Bullet Example

### Fragile version

```md
| Axis | Old | New |
|---|---|---|
| Inference | Multi-step | One-step |
| Signal | Score | Drift |
```

### Safer version

```md
- Old paradigm
  - inference: multi-step
  - training signal: score-based
- New paradigm
  - inference: one-step
  - training signal: drift-based
```

Use bullets when the table does not add much structure.

---

## 4. Claim Hygiene Example

### Better

- `The paper reports state-of-the-art 1-NFE FID on ImageNet 256x256 under its evaluated setting.`
- `This result is strong, but the evidence is still concentrated on ImageNet-scale image generation.`

### Worse

- `This paper completely changes generative modeling.`
- `It is obviously the new standard.`

Rule:

- strong words need scope
- scope needs evidence

---

## 5. Good Paper Info Format

```md
## Paper Info

- **Title:** Generative Modeling via Drifting
- **Authors:** Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He
- **Affiliations:** MIT, Harvard University
- **Venue:** arXiv preprint
- **Published:** 2026-02-06
- **Paper:** https://arxiv.org/abs/2602.04770
- **Project:** https://lambertae.github.io/projects/drifting/
```

Why this format is preferred:

- easier to scan
- less fragile than a Markdown table
- mobile-friendly

---

## 6. Architecture Coverage Example

### Too thin

```md
The model uses a transformer backbone and a diffusion head.
```

### Better

```md
The model has three major stages.

- An encoder turns raw inputs into token and pair representations.
- A trunk repeatedly updates those representations with pair-aware attention.
- A diffusion generator takes noisy coordinates and denoises them using the trunk context.

The key architectural change over the baseline is that structure generation is no longer a small prediction head. It becomes a full conditional coordinate generator.
```

Why the better version works:

- names the stages
- gives execution order
- says what each stage consumes and produces
- explains what changed relative to prior work

---

## 7. Safe Prompting Pattern

A stable internal instruction pattern is:

- write planning and protocol in English
- produce the final post in Korean
- prefer explanatory depth over summary
- include one representative local image
- explain architecture and pipeline explicitly
- avoid Mermaid unless clearly needed
- verify deploy visibility after push

---

## 8. Common Failure Modes

### Failure: post is not visible

Usually check in this order:

1. future publish date
2. deployment still running
3. broken front matter

### Failure: page renders awkwardly

Usually check:

1. wide tables
2. dense inline math
3. Mermaid overuse
4. missing spacing between blocks

### Failure: architecture section feels weak

Usually the issue is one of:

1. block names listed without explanation
2. pipeline order omitted
3. no input/output view of the trunk or generator
4. no pseudocode or implementation sketch
5. no comparison against the previous baseline architecture

### Failure: post feels shallow

Usually the issue is one of:

1. too much abstract paraphrase
2. not enough `How It Works`
3. results listed without interpretation
4. limitations too vague

---

## 9. Recommended Default Move

If uncertain, choose the safer path:

- bullets over tables
- display math over dense inline math
- real image over no image
- simpler structure over clever formatting
- narrower claim over overstated claim
- explicit architecture flow over vague model-name shorthand

---

## 10. Example Commands

Create a new scaffold:

```bash
python3 scripts/new_blog_post.py \
  --title "Paper Title" \
  --slug paper-title \
  --description "Short summary" \
  --category AI \
  --subcategory "Generative Models" \
  --tags tag1,tag2,tag3 \
  --paper-url https://arxiv.org/abs/0000.00000
```

Run common checks:

```bash
scripts/check_blog_post.sh _posts/YYYY-MM-DD-paper-title.md
```

Run rendered validation against live site:

```bash
python3 scripts/validate_blog_rendered.py --url https://eightmm.github.io --posts paper-title
```
