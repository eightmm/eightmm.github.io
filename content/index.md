---
title: Jaemin's Research Notes
---

# Jaemin's Research Notes

[GitHub](https://github.com/eightmm) · [eightmm.github.io](https://eightmm.github.io)

AI, computational biology, research infrastructure를 정리하는 개인 연구 블로그입니다. 글, 개념 노트, 논문 메모, 프로젝트 기록을 서로 연결해 장기적으로 다시 찾을 수 있게 만듭니다.

## Main Areas

| Area | Use For |
| --- | --- |
| [Agents](/agents) | LLM agent concepts, product features, tools, workflows, verification |
| [AI](/ai) | machine learning, architectures, learning methods, generative models, evaluation, systems |
| [Math](/math) | formulas and mathematical tools needed to read AI notes |
| [Computational Biology](/molecular-modeling) | focused notes on biological objects, sequence-based modeling, molecular and ligand modeling, interaction modeling, and structure-based workflows |
| [Infra](/infra) | hardware, GPU, HPC, storage, reproducibility, server operations |
| [Research](/research) | public research ideas, questions, hypotheses, and synthesis notes |
| [Projects](/projects) | public artifacts, implementations, design decisions, workflows |
| [Papers](/papers) | curated paper notes and reading utilities |
| [Posts](/posts) | reader-facing essays and topic maps |

## Content Groups

| Group | Area | 역할 |
| --- | --- | --- |
| Foundation | [AI](/ai), [Math](/math), [Computational Biology](/molecular-modeling), [Infra](/infra), [Agents](/agents) | 전반적인 설명과 오래 남는 개념 |
| My Work | [Research](/research), [Projects](/projects) | 내 연구 질문, 아이디어, 구현물, 운영한 workflow |
| Reading and Writing | [Papers](/papers), [Posts](/posts) | 논문 리뷰, 읽기 기록, 외부 독자를 위한 글 |

## Research와 Projects 구분

| Area | 중심 질문 | 예시 |
| --- | --- | --- |
| [Research](/research) | 어떤 공개 가능한 질문, 가설, 아이디어를 탐색하는가? | research direction, hypothesis, method comparison, experiment idea |
| [Projects](/projects) | 무엇을 실제로 만들고 운영하거나 공개 산출물로 남기는가? | pipeline, tool, blog/wiki system, runbook, reproducible artifact |

Research는 아이디어와 질문의 공간이고, Projects는 구현과 산출물의 공간입니다. 하나의 일이 둘 다 가질 수는 있지만, 페이지의 중심이 hypothesis면 Research, artifact면 Projects에 둡니다.

## 최근 글

- [[posts/2026-06-26-ai-wiki-map|AI Wiki를 어떤 축으로 나눠 볼 것인가]]
- [[posts/2026-06-25-blog-and-wiki-workflow|블로그와 위키를 같이 쓰는 방식]]
- [[posts/wiki-to-post-workflow|Wiki에서 Post로 승격하는 방식]]
- [[posts/2026-06-25-structure-based-ai-map|구조 기반 모델링을 어떻게 정리할 것인가]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
- [[posts/topic-roadmap|글감 로드맵]]

## 읽는 경로

- AI 기본기: [[math/index|Math]] -> [[ai/machine-learning|Machine learning]] -> [[ai/architectures|Architectures]] -> [[ai/learning-methods|Learning methods]] -> [[ai/evaluation|Evaluation]] -> [[ai/systems|Systems]]
- Protein sequence route: [[molecular-modeling/sequence-based|Sequence-based modeling]] -> [[molecular-modeling/proteins|Proteins]] -> [[concepts/protein-modeling/index|Protein modeling concepts]]
- Computational biology: [[molecular-modeling/index|Computational Biology]] -> [[molecular-modeling/entities|Objects and entities]] -> [[molecular-modeling/sequence-based|Sequence-based modeling]] -> [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]] -> [[molecular-modeling/interactions|Interaction modeling]] -> [[molecular-modeling/structure-based/index|Structure-based modeling]]
- Agents: [[agents/index|Agents]] -> [[agents/core/agent-loop|Agent loop]] -> [[agents/verification/verification-loop|Verification loop]]
- Infra: [[infra/index|Infra]] -> [[infra/hardware/memory-hierarchy|Memory hierarchy]] -> [[infra/hpc/job-lifecycle|HPC job lifecycle]] -> [[infra/reproducibility/run-record|Reproducible run record]]
- Papers: [[papers/index|Papers]] -> [[papers/workflows/paper-review-workflow|Paper review workflow]] -> [[papers/workflows/reading-status|Reading status]]
- Projects: [[projects/index|Projects]] -> [[projects/project-note-format|Project note format]] -> [[projects/llm-wiki-blog|LLM Wiki blog]]

## 운영 방식

글은 생각의 흐름을 정리하고, 개념·논문·모델 구조·평가 방법은 서로 연결된 노트로 축적합니다.

예를 들어 docking 글은 [[molecular-modeling/index|Computational Biology]]에서 시작해 [[molecular-modeling/structure-based/index|Structure-based modeling]], [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]], [[concepts/sbdd/scoring-function|Scoring function]], [[papers/sbdd/posebusters|PoseBusters]] 같은 내부 노트로 이어집니다.

작성 방식은 [[posts/blog-writing-guide|블로그 글 작성 가이드]]에 정리하고, 앞으로 풀어낼 글감은 [[posts/topic-roadmap|글감 로드맵]]에서 관리합니다.

Wiki note를 한글 글로 승격하는 기준은 [[posts/wiki-to-post-workflow|Wiki에서 Post로 승격하는 방식]]과 [[agents/workflows/content-promotion-workflow|Content promotion workflow]]에 둡니다.
