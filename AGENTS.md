# Repository Instructions

This repository is a Quartz v5 public research blog and LLM Wiki for `https://eightmm.github.io`.

## Scope

- Primary branch: `v5`.
- Main editable area for content work: `content/`.
- Do not edit `public/` directly. It is build output.
- Do not change `package.json`, lockfiles, `.github/workflows`, `quartz.config.yaml`, or dependencies unless the user explicitly asks.
- Run `npx quartz build` after content changes.

## Public Content Policy

Public pages must not include:

- Server IPs, hostnames, SSH ports, account names, usernames, private paths, credentials, tokens, keys, firewall details, or user lists.
- Internal task names, collaborator details, private datasets, unpublished experiment results, thesis-sensitive claims, or unreleased project details.
- Unverified DOI, arXiv IDs, paper metadata, metrics, or experimental results.

If a fact is missing from the provided source, write `to verify` instead of inventing it.

## Taxonomy

Keep the top-level site organized by broad axes:

- `entities/`: Protein, ligand, protein-ligand complex, and other modeled objects.
- `concepts/architectures/`: GNN, Transformer, state-space models, and model families.
- `concepts/geometric-deep-learning/`: Equivariance, geometry, coordinates, symmetry.
- `concepts/learning/`: SSL, JEPA, contrastive learning, and learning objectives.
- `agents/`: Agent workflows, coding agents, paper-ingestion agents, and verification habits.
- `research/`: Research domains and synthesis notes.
- `papers/`: Curated paper notes, not raw daily logs.
- `inbox/`: Sanitized daily paper briefs and uncurated candidates.
- `projects/`, `infra/`, `logs/`, `posts/`: Public project notes, infra notes, cleaned logs, and narrative posts.

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
