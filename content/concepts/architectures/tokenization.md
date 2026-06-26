---
title: Tokenization
tags:
  - architectures
  - representation-learning
---

# Tokenization

Tokenization chooses the basic units that a model receives. It is not only a text preprocessing step: residues, atoms, image patches, graph nodes, retrieved chunks, and tool calls can all be treated as tokens.

A sequence model usually starts from discrete tokens:

$$
t_1,\ldots,t_L = \operatorname{Tokenizer}(x)
$$

and maps them to vectors:

$$
X = [E_{t_1}+p_1,\ldots,E_{t_L}+p_L]^\top
$$

where $E_{t_i}$ is the embedding for token $t_i$, and $p_i$ is a positional or structural encoding.

## Common Choices

- Text: character, byte, subword, wordpiece, or sentence chunk.
- Image: patch, region, pixel, feature-map cell, or visual token.
- Video and audio: frame, clip, spectrogram patch, waveform chunk, or learned acoustic token.
- Protein sequence: amino acid residue, motif, span, or multiple-sequence-alignment token.
- Molecule: SMILES token, atom, bond-centered token, substructure, or conformer token.
- Grid or structure: voxel, contact-map cell, spatial window, or local coordinate token.
- Agent context: message, retrieved document chunk, tool call, observation, or memory item.

## Tokenization Contract

| Field | Question | Example |
| --- | --- | --- |
| Unit | What object becomes one token? | byte, subword, residue, atom, patch, node |
| Vocabulary | Is the token space fixed, learned, open, or continuous? | BPE vocab, amino acids, atom types |
| Boundary | How are sequence breaks, chains, documents, or components marked? | BOS/EOS, chain id, segment id |
| Length | What controls token count? | sequence length, image resolution, atom count |
| Lost information | What is discarded before modeling? | stereochemistry, whitespace, coordinates, metadata |
| Reversibility | Can outputs be decoded back to valid objects? | valid text, molecule, graph, action |

Tokenization is part of the model claim because it determines what the model can possibly learn.

## Scaling

For Transformer-style attention, token count directly affects cost:

$$
\mathrm{cost} \sim O(L^2d)
$$

where $L$ is token length and $d$ is hidden width. A paper claiming better architecture or training can actually be benefiting from shorter or more informative tokens.

## Domain-Specific Risks

| Domain | Token Risk |
| --- | --- |
| text | tokenizer changes can alter benchmark prompts and likelihoods |
| image | patch size trades spatial detail for sequence length |
| protein | residue tokens may miss MSA, domain, or structure context |
| molecule SMILES | different valid strings can represent the same molecule |
| molecular graph | atom/bond features depend on standardization and chemical state |
| structure | coordinate tokens need frames, units, and equivariance policy |
| agents | tool-call tokens need schema validity and execution feedback |

## Output Decoding

If the model generates tokens, validity depends on a decoder:

$$
\hat{x}
=
\operatorname{Detokenizer}(\hat{t}_1,\ldots,\hat{t}_L)
$$

For structured objects, detokenization can fail. Report invalid strings, invalid graphs, schema violations, or failed tool calls instead of silently dropping them.

## Checks

- What information is lost before the model sees the input?
- Does token length scale with sequence length, atom count, image resolution, or retrieved context size?
- Are positional, chain, segment, or structural encodings needed?
- Does tokenization leak labels, future information, or evaluation-set-specific preprocessing?
- Is the tokenizer trained on test or future data?
- Are invalid detokenized outputs counted in evaluation?

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/modalities/representation-contract|Representation contract]]
