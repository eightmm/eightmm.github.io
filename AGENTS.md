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

Keep the top-level site organized by broad axes:

- `entities/`: Protein, ligand, molecule, protein-ligand complex, sequence, structure, assay, dataset, genome, and other directly modeled objects.
- `concepts/architectures/`: GNN, Transformer, state-space models, and model families.
- `concepts/geometric-deep-learning/`: Equivariance, geometry, coordinates, symmetry.
- `concepts/learning/`: SSL, JEPA, contrastive learning, and learning objectives.
- `agents/`: Agent workflows, coding agents, paper-ingestion agents, and verification habits.
- `research/`: Research domains and synthesis notes.
- `papers/`: Curated paper notes, not raw daily logs.
- `inbox/`: Sanitized daily paper briefs and uncurated candidates.
- `projects/`, `infra/`, `logs/`, `posts/`: Public project notes, infra notes, cleaned logs, and narrative posts.

Bio scope should stay focused on structure-based AI, protein modeling, ligand/molecule modeling, protein-ligand interaction, and genome/sequence modeling. Do not open broad omics, transcriptomics, single-cell, pathway biology, clinical omics, or systems biology unless the user explicitly expands the scope.

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
