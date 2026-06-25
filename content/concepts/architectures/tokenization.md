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

## Checks

- What information is lost before the model sees the input?
- Does token length scale with sequence length, atom count, image resolution, or retrieved context size?
- Are positional, chain, segment, or structural encodings needed?
- Does tokenization leak labels, future information, or evaluation-set-specific preprocessing?

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
