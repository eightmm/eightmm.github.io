---
title: Molexar — Unified Multimodal Molecular Foundation Model for Drug Design
aliases:
  - papers/molexar
tags:
  - paper
  - molecular-generation
  - generative-models
  - structure-based-modeling
status: reading
source_type: ArXiv
source_url: https://arxiv.org/abs/2606.25865
---

# Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design

## 한 줄 요약

Fragment-SELFIES를 기반으로 scalar property, pharmacophore fingerprint, protein sequence, binding pocket 조건을 하나의 autoregressive generation 경로로 다루려는 multimodal molecular foundation model입니다.

## 왜 중요한가

이 논문은 [[molecular-modeling/structure-based/index|Structure-based modeling]]과 [[concepts/generative-models/molecular-generation|molecular generation]]을 연결해서 읽을 수 있습니다. 조건마다 별도 head를 두는 대신 single-path design을 쓰기 때문에 multi-property drug design pipeline을 단순화할 가능성이 있습니다.

## 핵심 포인트

- Fragment-SELFIES는 fragment-aware molecular language이며 validity-preserving decoding을 목표로 합니다.
- Single autoregressive decoder가 value-token embedding의 in-place replacement를 통해 여러 condition type을 처리합니다.
- Pretrained model에서는 unconditional 및 fragment-constrained generation에서 높은 validity와 drug-likeness를 보고합니다.
- SFT에서는 single-property와 multi-property instruction following을 보고합니다.
- CrossDocked2020 기반 target-conditioned generation 결과를 보고합니다.
- MolGenBench safety와 potency 결과를 보고하지만, benchmark detail은 추가 확인이 필요합니다.

## 읽을 때 볼 질문

- Fragment-SELFIES는 syntax constraint와 coverage 측면에서 standard SELFIES와 어떻게 다른가?
- In-place embedding replacement가 condition type 사이의 interference를 만들지 않는가?
- Property type별 validity distribution은 어떤가? Multi-property conditioning이 single-property performance를 떨어뜨리지는 않는가?
- Specialized per-property model과 비교할 때 parameter count 증가에 따른 scaling은 어떤가?
- Target-conditioned result가 ligand scaffold split, protein-family split, pocket similarity leakage에 대해 robust한가?

## Artifact 공개 상태

| Artifact | Status | Notes |
|---|---|---|
| Paper | found | arXiv abstract page |
| Code | to verify | project page 또는 linked repository 확인 필요 |
| Data | to verify | CrossDocked2020 및 MolGenBench protocol detail 확인 필요 |
| Splits | to verify | target-conditioned generation split 설명 확인 필요 |
| Weights | to verify | public checkpoint 상태 미확인 |
| Environment | to verify | implementation dependency 미확인 |

## 관련 노트

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/fragment-selfies|Fragment-SELFIES]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]

## Metadata

- arXiv: [2606.25865](https://arxiv.org/abs/2606.25865)
- Submitted: 2026-06-24
- Authors: Haoyu Lin, Yiyan Liao, Jinmei Pan, Xinliao Ling, Luhua Lai, Jianfeng Pei
- Category: q-bio.BM
