---
title: Multi-Scale ML for Antibody-Antigen Binding Affinity Prediction
tags:
  - paper
  - binding-affinity
  - protein-modeling
  - evaluation
status: reading
source_type: bioRxiv
---

# Multi-Scale Machine Learning for Antibody-Antigen Binding Affinity Prediction

## One-line Summary

A reported multi-scale ML framework for antibody-antigen binding affinity prediction using deep mutational scanning and structural features. Detailed metrics should remain `to verify` until the paper is read directly.

## Why It Matters

This paper may be relevant to [[research/structure-based-ai/index|Structure-based AI]] and [[concepts/sbdd/binding-affinity|binding affinity]]. Its useful angle is not only model performance, but how antibody-antigen binding evaluation handles mutation units, pathogen or antigen holdouts, confidence filtering, and theoretical performance ceilings.

## Key Points

- Reported multi-scale descriptor set spanning sequence, structure, physicochemical, and learned representation features
- Reported use of deep mutational scanning and structural features
- Reported confidence-stratified prediction; exact coverage/metric values to verify
- Reported cross-pathogen or cross-antigen transfer limitations; split protocol to verify
- [[concepts/evaluation/boltzmann-ceiling|Boltzmann ceiling analysis]] appears to be a central evaluation idea; formula and assumptions need direct reading

## Reading Questions

- What is the variance across pathogens in the MCC distribution?
- Does the confidence stratification primarily reduce false positives or increase true positives?
- How does the ESM-2 embedding modality compare to structural features alone?
- What would be needed to close the gap from 79.1% to 100% of the Boltzmann ceiling?
- What exactly is the prediction unit: mutation, antibody-antigen pair, antigen family, or assay row?
- Are all measurements from one DMS experiment kept on the same side of each split?

## Artifact Availability

| Artifact | Status | Notes |
|---|---|---|
| Paper | found | bioRxiv preprint page |
| Code | to verify | Search linked repository or supplementary materials |
| Data | to verify | AbAgym/DMS source and filtering rules |
| Splits | to verify | Need LOCO-DMS or holdout definition |
| Weights | not applicable | Classical ML or ensemble status to verify |
| Environment | to verify | Feature extraction versions unknown |

## Related Notes

- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/boltzmann-ceiling|Boltzmann ceiling analysis]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[papers/artifact-availability|Artifact availability]]

## Metadata

- DOI: verified on bioRxiv
- Venue: bioRxiv
- Posted: 2026-06-23
- Authors: S. Sivasubramani
