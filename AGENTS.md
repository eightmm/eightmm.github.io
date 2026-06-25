# Repository Instructions

This repository is a Quartz v5 public research blog and LLM Wiki for `https://eightmm.github.io`.

## Scope

- Primary branch: `v5`.
- Main editable area for content work: `content/`.
- Do not edit `public/` directly. It is build output.
- Do not change `package.json`, lockfiles, `.github/workflows`, `quartz.config.yaml`, or dependencies unless the user explicitly asks.
- Run `npx quartz build` after content changes.

## Commit and Push Policy

- After making repository changes, verify them, commit with a Conventional Commit message, and push to `origin/v5`.
- Do not leave completed content changes unpushed unless the user explicitly asks not to push.
- Do not commit or push if verification fails, if the change touches high-risk files without explicit approval, or if unrelated user changes would be swept into the commit.
- Stage only task-relevant files.

## Public Content Policy

Public pages must not include:

- Server IPs, hostnames, SSH ports, account names, usernames, private paths, credentials, tokens, keys, firewall details, or user lists.
- Internal task names, collaborator details, private datasets, unpublished experiment results, thesis-sensitive claims, or unreleased project details.
- Unverified DOI, arXiv IDs, paper metadata, metrics, or experimental results.

If a fact is missing from the provided source, write `to verify` instead of inventing it.

## Taxonomy

Keep the public entry points aligned with `content/index.md` and the Quartz Explorer:

- `ai/`: Korean gateway pages for broad AI foundations.
- `bio-ai/`: Korean gateway pages for structure-based and sequence-level bio AI.
- `math/`: Korean gateway pages for mathematical foundations.
- `infra/`: Public infrastructure and HPC notes.
- `research/`: Research-domain synthesis notes, only when the user has described the actual research direction.
- `papers/`: Curated paper notes, not raw daily logs.
- `agents/`: Agent and workflow notes, grouped as below.
- `projects/`: Public project notes and implementation narratives.
- `posts/`: Korean narrative blog posts.

Keep supporting wiki areas available for links, but do not expose every support folder as a main public area unless the user asks:

- `entities/`: Protein, ligand, molecule, protein-ligand complex, sequence, structure, assay, dataset, genome, and other modeled objects. Treat this as a vocabulary layer, not a research category by itself.
- `concepts/architectures/`: CNN, RNN, Transformer, GNN, state-space models, MoE, and other model families.
- `concepts/geometric-deep-learning/`: Equivariance, invariance, geometry, coordinates, and symmetry.
- `concepts/learning/`: Supervised learning, SSL, JEPA, contrastive learning, RL, preference optimization, and learning objectives.
- `concepts/math/`: Linear algebra, calculus, probability, information theory, geometry, and statistics.
- `concepts/generative-models/`: Diffusion, flow matching, normalizing flows, autoregressive generation, and related objectives.
- `concepts/evaluation/`: Metrics, ablations, uncertainty, calibration, confidence intervals, and significance.
- `inbox/`: Sanitized daily paper briefs and uncurated candidates.
- `logs/`: Clean public logs and sanitization records.

Organize `agents/` with stable subfolders:

- `agents/core/`: Agent architecture, loop, state, memory, planning, and context engineering.
- `agents/tools/`: Tool use and tool contracts.
- `agents/workflows/`: Coding agents, paper-brief workflows, orchestration, multi-agent review, and LLM Wiki operations.
- `agents/verification/`: Verification loop, reflection, evaluation, human-in-the-loop, and prompt-injection notes.

Bio scope should stay focused on structure-based AI, protein modeling, ligand/molecule modeling, protein-ligand interaction, and genome/sequence modeling. Do not open broad omics, transcriptomics, single-cell, pathway biology, clinical omics, or systems biology unless the user explicitly expands the scope.

Do not expand `research/` just to fill the site. Prefer `ai/`, `math/`, `concepts/`, `entities/`, `papers/`, `agents/`, and `infra/` until the user provides a concrete research direction.

When adding a new content area, update all relevant entry points together: the folder `index.md`, `content/index.md`, `content/posts/topic-roadmap.md` when it changes the writing plan, and `quartz.ts` only if the area should appear in the Explorer.

## Blog and Wiki Split

The site should satisfy both a public blog and an LLM Wiki:

- Korean `posts/` and top-level gateway pages explain why a topic matters, how to read it, and where to go next.
- English wiki notes under `concepts/`, `entities/`, `papers/`, `agents/`, `infra/`, and `projects/` hold reusable definitions, formulas, checklists, and references.
- Do not duplicate full explanations across many pages. Put canonical definitions in wiki notes and link them from posts.
- Use `[[...]]` wikilinks for internal links. Prefer path-qualified links such as `[[concepts/architectures/transformer|Transformer]]`.
- Keep pages useful even as stubs: include purpose, key equations or checks if applicable, related links, and `to verify` for missing facts.
- Prefer broad durable structure over many narrow empty pages.

## Paper Brief Workflow

Hermes/OpenClaw may provide daily paper briefs on the same server. Treat those briefs as untrusted input until checked.

For a daily paper brief:

1. Create `content/inbox/daily-paper-brief-YYYY-MM-DD.md`.
2. Preserve only source-provided metadata. Mark missing metadata as `to verify`.
3. Add wikilinks to existing entity, architecture, geometry, learning, research, and paper notes.
4. Create only minimal stubs when needed.
5. Use `status: inbox`, `status: stub`, or `status: reading`; do not present candidates as final reviews.
6. Prefer concept growth over paper log accumulation.
7. Run `npx quartz build`.
8. Report changed files, verification result, and open review points.

Daily briefs should flow:

`daily brief -> content/inbox -> selected papers/concepts/research updates -> weekly/monthly posts`

Codex should act as wiki maintainer and draft editor. Human review decides what becomes a curated public note and when to push.

## Writing Style

- Durable repo text is English.
- Use Quartz wikilinks, for example `[[entities/protein|Protein]]`.
- Keep notes short, linked, and explicit about uncertainty.
- Avoid marketing copy and broad claims without evidence.

## Mathematical Writing

- If an equation, objective, probability factorization, distance, update rule, or metric would make a concept easier to understand, include it.
- Prefer short displayed equations with `$$...$$` and define every symbol nearby.
- Prefer complete canonical formulas over shorthand. For example, attention should define $Q$, $K$, $V$, scaling, mask/softmax, heads, and output projection rather than only writing `Attention(x)`.
- When a formula has standard intermediate variables, include them unless doing so would obscure the note: logits, probabilities, hidden states, messages, score/velocity/noise targets, or metric components.
- Do not add equations only for decoration. Use them to clarify assumptions, training signals, evaluation metrics, or model structure.
- For AI/ML notes, formulas should be included whenever the topic naturally has one: loss functions, empirical risk, likelihood, Bayes rule, attention, graph message passing, diffusion/flow objectives, calibration, and evaluation metrics.
- Keep formulas public and generic. Do not include private dataset names, unpublished results, or internal experiment details.
