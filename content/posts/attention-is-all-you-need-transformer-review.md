---
title: Attention Is All You Need를 지금 다시 읽는 법
date: 2026-06-30
tags:
  - posts
  - ai
  - transformer
  - papers
---

# Attention Is All You Need를 지금 다시 읽는 법

`Attention Is All You Need`는 Transformer를 제안한 논문입니다. 그런데 지금 이 논문을 읽을 때는 조심해야 합니다. 지금의 LLM 시대를 이미 알고 읽으면, 이 논문이 실제로 증명한 것보다 훨씬 큰 이야기를 했다고 착각하기 쉽습니다.

이 글의 목표는 논문을 번역하듯 따라가는 것이 아니라, “이 논문이 어떤 문제를 바꿨고, Transformer라는 구조를 어떻게 이해해야 하는가”를 정리하는 것입니다.

짧은 논문 카드는 [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]에 따로 두고, 이 글은 한글 longform 해설로 둡니다.

## 먼저 볼 지도

| 질문 | 이 글에서 보는 것 | 더 볼 곳 |
| --- | --- | --- |
| 이 논문이 바꾼 문제는 무엇인가? | recurrent/convolution 기반 sequence transduction에서 attention 중심 구조로 이동 | [[concepts/architectures/transformer|Transformer]] |
| Attention은 무슨 계산인가? | query, key, value, softmax, weighted sum | [[concepts/architectures/attention|Attention]] |
| Transformer는 attention 하나인가? | attention, FFN, residual, normalization, positional encoding의 조합 | [[ai/architectures|AI architectures]] |
| 논문이 실제로 증명한 것은? | WMT translation setting에서 강한 성능과 병렬성 | [[papers/analysis/benchmark-card|Benchmark card]] |
| 논문이 증명하지 않은 것은? | 현대 LLM 전체, long-context 일반화, attention 설명가능성 | [[papers/analysis/limitation-taxonomy|Limitation taxonomy]] |

## 논문을 읽기 전 배경

Transformer 이전의 강한 sequence model은 보통 순서 정보를 직접 따라가는 구조였습니다.

RNN이나 LSTM은 이전 hidden state를 받아 다음 hidden state를 만듭니다.

$$
h_t = f(h_{t-1}, x_t)
$$

이 구조는 자연스럽습니다. 문장은 순서가 있고, 단백질 sequence도 순서가 있고, time series도 순서가 있습니다. 그래서 이전 상태를 누적하면서 다음 상태를 만드는 방식은 강한 inductive bias를 가집니다.

하지만 단점도 명확합니다. $h_t$를 계산하려면 $h_{t-1}$이 필요합니다. 따라서 sequence length 방향으로 완전히 병렬화하기 어렵습니다.

CNN 기반 sequence model은 local window를 보면서 점점 receptive field를 넓힙니다. 이 방식은 병렬화가 좋지만, 멀리 떨어진 token 사이의 관계를 직접 연결하려면 깊이, dilation, 여러 layer가 필요합니다.

Attention은 질문을 다르게 던집니다.

> 순서대로 상태를 넘기지 말고, 각 token이 필요한 다른 token을 직접 참고하면 안 되는가?

이 질문이 Transformer의 출발점입니다.

여기서 중요한 것은 “순서 정보를 버렸다”가 아닙니다. Transformer는 순서를 무시한 것이 아니라, 순서 처리와 token interaction을 분리했습니다. RNN은 순서 처리와 정보 전달이 같은 recurrence 안에 묶여 있습니다. Transformer는 token끼리 정보를 섞는 일은 attention이 맡고, 순서 정보는 positional encoding이 따로 제공합니다.

이 분리가 큰 전환이었습니다.

| 관점 | RNN 계열 | Transformer 계열 |
| --- | --- | --- |
| token 간 정보 전달 | hidden state를 순차적으로 전달 | attention matrix로 직접 연결 |
| 병렬성 | sequence 방향 의존성이 큼 | layer 안에서 token 병렬 계산 가능 |
| 순서 정보 | recurrence 자체에 내장 | positional encoding으로 주입 |
| 긴 거리 관계 | 여러 step을 지나 전달 | 한 attention layer에서 직접 연결 가능 |
| 비용 병목 | sequential dependency | dense attention의 $T^2$ 비용 |

그래서 이 논문은 “attention이라는 새 module”보다 “sequence model을 구성하는 방식”을 바꾼 논문으로 보는 편이 좋습니다.

## 핵심 아이디어

Transformer의 핵심은 다음처럼 요약할 수 있습니다.

$$
\text{sequence modeling}
\approx
\text{attention-based token mixing}
+
\text{position-wise nonlinear processing}
+
\text{positional information}
$$

즉, Transformer는 attention 하나가 아닙니다. Transformer block은 여러 구성요소의 조합입니다.

| 구성요소 | 역할 |
| --- | --- |
| Multi-head attention | token 사이 정보를 섞음 |
| Feed-forward network | 각 token representation을 비선형 변환 |
| Residual connection | 깊은 layer 학습을 안정화 |
| Normalization | activation scale과 training 안정성 조절 |
| Positional encoding | 순서 정보를 주입 |
| Mask | 어떤 token을 볼 수 있는지 제한 |

논문 제목은 “Attention Is All You Need”지만, 실제 모델은 attention만으로 이루어진 수식 하나가 아닙니다. attention을 중심으로 한 전체 architecture recipe입니다.

## Attention 수식

논문에서 가장 중요한 수식은 scaled dot-product attention입니다.

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

각 기호는 다음을 뜻합니다.

| 기호 | 의미 |
| --- | --- |
| $Q$ | query |
| $K$ | key |
| $V$ | value |
| $d_k$ | key/query dimension |
| $QK^\top$ | query와 key 사이의 모든 dot product |
| $\sqrt{d_k}$ | logit scale 조절 |
| softmax | key 방향으로 normalize된 attention weight |

한 query $q_i$가 key $k_j$에 주는 attention weight는 다음처럼 볼 수 있습니다.

$$
\alpha_{ij}
=
\frac{
\exp(q_i^\top k_j / \sqrt{d_k})
}{
\sum_{\ell}
\exp(q_i^\top k_\ell / \sqrt{d_k})
}
$$

그리고 output은 value의 weighted sum입니다.

$$
y_i
=
\sum_j \alpha_{ij}v_j
$$

직관적으로는 이렇습니다.

| 요소 | 직관 |
| --- | --- |
| query | 지금 token이 찾고 싶은 정보 |
| key | 각 token이 “나는 이런 정보와 match된다”고 내놓는 주소 |
| value | 실제로 전달되는 내용 |
| softmax weight | 어느 token을 얼마나 참고할지 |

Attention은 “비슷한 token을 찾는다” 정도로만 이해하면 부족합니다. Query-key similarity는 어디를 볼지 정하고, value는 실제로 섞이는 정보를 담습니다. 그래서 attention weight만 보고 모델의 결정을 설명했다고 말하면 위험합니다. 이후 layer, residual path, value vector, FFN이 모두 최종 출력을 바꿉니다.

## 왜 $\sqrt{d_k}$로 나누는가

query와 key의 dimension이 커지면 dot product의 크기도 커지기 쉽습니다. logit이 너무 커지면 softmax가 한쪽으로 날카로워지고, gradient가 불안정해질 수 있습니다.

그래서 논문은 dot product를 $\sqrt{d_k}$로 나눕니다.

$$
\frac{q^\top k}{\sqrt{d_k}}
$$

이 부분은 모델링 아이디어와 numerical stability가 만나는 지점입니다. Transformer를 이해할 때는 이런 작은 scaling trick도 중요합니다. architecture paper의 성능은 보통 “큰 아이디어” 하나만으로 나오지 않고, 학습이 잘 되도록 만드는 세부 설계와 함께 나옵니다.

## Multi-head attention

논문은 하나의 attention만 쓰지 않고 여러 head를 병렬로 사용합니다.

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

각 head는 서로 다른 projection을 사용합니다. 그래서 같은 token sequence라도 여러 subspace에서 관계를 볼 수 있습니다.

하지만 “head 1은 문법, head 2는 의미, head 3은 장거리 의존성”처럼 단정하면 안 됩니다. 그렇게 해석될 수 있는 head가 있을 수는 있지만, head의 역할은 학습 결과이고 항상 사람이 해석하기 쉽게 분리되지는 않습니다.

더 안전한 해석은 이렇습니다.

> Multi-head attention은 하나의 similarity space에 모든 관계를 밀어 넣지 않고, 여러 projection space에서 token mixing을 수행하게 해준다.

Multi-head attention을 shape 관점에서 보면 더 명확합니다. 입력 token representation을 $X\in\mathbb{R}^{T\times d_{\mathrm{model}}}$라고 하면 각 head는 서로 다른 projection을 만듭니다.

$$
Q_i=XW_i^Q,\quad K_i=XW_i^K,\quad V_i=XW_i^V
$$

각 head는 다음 shape의 output을 만듭니다.

$$
\operatorname{head}_i\in\mathbb{R}^{T\times d_v}
$$

여러 head를 concat하면:

$$
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)
\in
\mathbb{R}^{T\times hd_v}
$$

마지막 output projection $W^O$가 이를 다시 model dimension으로 보냅니다.

이 구조 덕분에 attention은 단일 relation matrix가 아니라 여러 relation view를 동시에 학습할 수 있습니다. 다만 head 수를 늘린다고 무조건 좋아지는 것은 아닙니다. Head dimension, total parameter count, training budget, data size가 함께 바뀌기 때문입니다. 그래서 논문에서 head 수와 dimension을 ablation한 부분은 architecture claim을 읽을 때 중요한 근거입니다.

## Encoder와 decoder

원 논문의 Transformer는 encoder-decoder 구조입니다.

Encoder는 source sentence를 읽습니다. Decoder는 target sentence를 생성합니다. Decoder는 두 종류의 attention을 사용합니다.

| Attention type | Query | Key/value | 역할 |
| --- | --- | --- | --- |
| encoder self-attention | source token | source token | source representation 생성 |
| decoder masked self-attention | target prefix | target prefix | 이전 target token만 보고 생성 |
| encoder-decoder attention | decoder state | encoder output | source sentence 참고 |

Decoder self-attention에는 mask가 들어갑니다.

$$
\operatorname{Attention}(Q,K,V,M)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

여기서 $M$은 미래 token을 못 보게 막는 역할을 합니다. Autoregressive generation에서는 position $t$가 $t+1$ 이후 token을 보면 안 됩니다.

이 점이 중요합니다. 지금 우리가 흔히 말하는 decoder-only LLM은 이 논문에서 사용한 encoder-decoder Transformer와 같지 않습니다. 같은 Transformer family 안에 있지만, 구조와 학습 목적이 다릅니다.

## Positional encoding

Attention만 있으면 token 순서를 모릅니다. Self-attention은 token들 사이의 pairwise interaction을 계산하지만, 입력 순서 자체를 자동으로 알지는 못합니다.

그래서 논문은 sinusoidal positional encoding을 사용합니다.

$$
\operatorname{PE}_{(pos,2i)}
=
\sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

$$
\operatorname{PE}_{(pos,2i+1)}
=
\cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

그리고 token embedding에 더합니다.

$$
x_t = e_t + \operatorname{PE}_t
$$

즉, token의 의미 정보 $e_t$에 위치 정보 $\operatorname{PE}_t$를 섞어 넣습니다.

이 논문을 지금 읽을 때는 positional encoding을 너무 단순하게 보면 안 됩니다. 이후 Transformer 계열에서는 learned absolute position, relative position bias, rotary position embedding, ALiBi, 구조적 position encoding 등 많은 변형이 나왔습니다.

그래도 원 논문의 핵심 메시지는 여전히 중요합니다.

> Attention 기반 모델은 순서나 위치에 대한 정보를 별도로 설계해야 한다.

이건 단백질 sequence, molecule graph, protein-ligand complex, document chunk, tool-call sequence를 다룰 때도 반복해서 등장합니다.

예를 들어 단백질 sequence에서는 residue index가 중요하지만, structure-aware model에서는 3D 거리나 local frame이 더 중요할 수 있습니다. 문서 검색에서는 chunk order와 source document가 중요할 수 있고, agent workflow에서는 message order와 tool-call dependency가 중요할 수 있습니다.

그래서 positional encoding은 단순히 “몇 번째 token인가”의 문제가 아닙니다.

| 대상 | position이 의미하는 것 |
| --- | --- |
| 자연어 문장 | token 순서 |
| protein sequence | residue index, chain boundary |
| molecule graph | atom order가 아니라 graph distance나 bond relation |
| protein structure | coordinate, residue index, local frame |
| retrieved document | chunk order, document source, citation span |
| agent trace | message order, tool call order, dependency |

이 관점에서 보면 Transformer는 domain마다 position contract를 다시 써야 하는 구조입니다. 그냥 token만 넣는다고 항상 올바른 inductive bias가 생기지는 않습니다.

## Feed-forward network

Attention은 token 사이 정보를 섞습니다. 하지만 섞기만 해서는 충분하지 않습니다. 각 token representation을 비선형적으로 변환하는 block이 필요합니다.

논문은 position-wise feed-forward network를 사용합니다.

$$
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
$$

이 FFN은 각 position에 독립적으로 적용됩니다. 즉, token 간 interaction은 attention에서 일어나고, FFN은 각 token의 representation을 변환합니다.

현대 Transformer를 볼 때도 이 구분은 유용합니다.

| Block | 역할 |
| --- | --- |
| Attention | token 간 정보 이동 |
| MLP/FFN | token별 feature 변환 |

그래서 Transformer를 단순히 attention model이라고 부르면 반만 맞습니다. Transformer는 attention과 MLP가 번갈아 쌓인 구조입니다.

## 계산 복잡도와 병렬성

Transformer의 강점 중 하나는 sequence position 방향 병렬성입니다. RNN은 $h_t$가 $h_{t-1}$에 의존하지만, self-attention은 한 layer 안에서 모든 token pair score를 행렬곱으로 계산할 수 있습니다.

하지만 대가가 있습니다.

Sequence length가 $T$일 때 dense self-attention은 $T\times T$ score matrix를 만듭니다.

$$
S = QK^\top
$$

attention weight memory는 대략:

$$
O(T^2)
$$

score 계산은 대략:

$$
O(T^2d_k)
$$

그래서 Transformer는 병렬성은 좋지만 긴 context에서는 부담이 커집니다. 이 문제 때문에 나중에 sparse attention, local attention, state-space model, Mamba, retrieval, chunking, KV cache, long-context kernel 같은 많은 흐름이 나옵니다.

원 논문의 효율성 claim은 당시 translation setting과 hardware context 안에서 읽어야 합니다. “Transformer는 항상 효율적이다”가 아니라, “recurrent dependency를 제거해 병렬화에 유리한 sequence transduction model을 만들었다”가 더 정확합니다.

## 논문이 보여준 증거

주요 실험은 WMT 2014 English-German, English-French machine translation입니다. 논문은 Transformer가 강한 BLEU score를 내고, training cost 측면에서도 경쟁력이 있음을 보입니다. 추가로 constituency parsing 실험도 포함해 translation에만 완전히 갇힌 구조가 아님을 보여줍니다.

증거를 claim별로 나누면 이렇습니다.

| Claim | Evidence | 해석 |
| --- | --- | --- |
| attention-only encoder-decoder가 강하다 | WMT translation 결과 | sequence transduction setting에서 강한 architecture |
| recurrence 없이 병렬 학습이 가능하다 | architecture와 training cost 비교 | position-wise recurrence 제거의 실용적 장점 |
| multi-head attention이 의미 있다 | head 수, dimension, model size ablation | component-level evidence |
| positional encoding이 필요하다 | sinusoidal/learned positional encoding 비교 | order signal이 필요함 |
| translation 밖으로도 가능성이 있다 | parsing 실험 | broad transfer의 초기 근거이지만 제한적 |

여기서 중요한 것은 evidence boundary입니다. 논문이 보여준 것은 매우 강력하지만, 그 범위는 translation 중심 benchmark입니다. 현대 LLM, tool-use agent, retrieval-augmented generation, protein modeling까지 직접 증명한 것은 아닙니다.

논문을 읽을 때는 result table의 숫자를 외우는 것보다, 어떤 claim을 어떤 evidence가 지지하는지 분리하는 것이 더 중요합니다.

| 읽을 항목 | 질문 |
| --- | --- |
| benchmark | 어떤 task와 split에서 평가했는가? |
| metric | BLEU가 실제로 무엇을 측정하는가? |
| baseline | 당시 강한 recurrent/convolutional baseline과 비교했는가? |
| ablation | 어떤 component의 필요성을 분리해서 보였는가? |
| training cost | architecture 때문인가, implementation/hardware 때문인가? |
| transfer | parsing 실험이 어디까지 일반화를 지지하는가? |

이런 식으로 읽으면 “Transformer가 모든 것을 이겼다”가 아니라, “Transformer가 어떤 조건에서 무엇을 강하게 보였는가”가 남습니다. 이 차이가 papers를 자산으로 쌓을 때 중요합니다.

## 논문이 증명하지 않은 것

이 논문이 유명해지면서 과장된 해석도 많아졌습니다.

| 흔한 오해 | 더 정확한 해석 |
| --- | --- |
| 이 논문이 현대 LLM을 바로 제안했다 | Transformer architecture를 sequence transduction setting에서 제안했다 |
| attention은 언제나 recurrence보다 낫다 | 논문 setting에서는 recurrence 없이도 강한 성능을 보였다 |
| attention weight는 설명가능성이다 | attention weight는 intermediate mixing coefficient일 뿐이다 |
| sinusoidal position이면 long context가 해결된다 | extrapolation 가능성이 있지만 실제 성능은 별도 검증이 필요하다 |
| 원 논문 구조가 지금 표준 Transformer와 같다 | 이후 norm placement, activation, position encoding, objective 등이 많이 바뀌었다 |

논문을 자산으로 쌓을 때는 이런 구분이 중요합니다. Paper note는 paper가 실제로 보인 claim을 보존해야 하고, 나중에 생긴 역사적 의미는 별도로 연결해야 합니다.

## 지금 이 논문을 읽는 방법

지금 읽을 때는 네 층을 분리하면 좋습니다.

| 층 | 원 논문 | 이후 확장 |
| --- | --- | --- |
| Architecture | encoder-decoder Transformer | encoder-only, decoder-only, multimodal, graph, protein, vision Transformer |
| Objective | supervised translation likelihood | language modeling, masked modeling, instruction tuning, preference optimization |
| Data | WMT translation, parsing | web-scale text, code, multimodal data, scientific sequence |
| Systems | 병렬 학습이 유리한 구조 | distributed training, KV cache, long-context serving |

Transformer가 현대 AI의 중심이 된 것은 이 네 층이 모두 확장됐기 때문입니다. 원 논문 하나가 모든 것을 해결한 것이 아니라, architecture family가 이후 data, objective, scale, systems와 결합하면서 커진 것입니다.

## 구현 관점 체크리스트

이 논문을 보고 Transformer를 구현하거나, 다른 Transformer 논문을 읽을 때는 다음을 확인해야 합니다.

| 항목 | 왜 중요한가 |
| --- | --- |
| normalization placement | 원 논문 post-norm과 현대 pre-norm은 학습 안정성이 다름 |
| mask convention | causal/padding/cross-attention mask의 axis가 중요 |
| positional encoding | absolute, relative, rotary 등에 따라 extrapolation 성질이 다름 |
| label smoothing | loss와 calibration, BLEU에 영향을 줌 |
| learning-rate schedule | warmup과 decay가 training recipe의 일부 |
| weight sharing | embedding/output projection parameterization이 달라짐 |
| decoding | beam search, length penalty가 translation score에 영향을 줌 |

Architecture diagram만 보고 같은 Transformer라고 생각하면 안 됩니다. Training recipe와 evaluation protocol까지 포함해야 논문 결과를 이해할 수 있습니다.

특히 원 논문과 현대 구현 사이에는 꽤 많은 차이가 있습니다.

| 항목 | 원 논문을 읽을 때 | 현대 구현을 볼 때 |
| --- | --- | --- |
| norm placement | post-norm 계열로 읽기 | pre-norm인지 확인 |
| activation | ReLU FFN | GELU, SwiGLU 등 확인 |
| position | sinusoidal or learned | RoPE, relative bias, ALiBi 등 확인 |
| objective | translation likelihood | LM, masked LM, instruction tuning 등 확인 |
| architecture | encoder-decoder | encoder-only, decoder-only, hybrid 확인 |
| inference | translation decoding | KV cache, sampling, tool use 등 확인 |

그래서 “Transformer를 안다”는 말은 너무 넓습니다. 최소한 attention, mask, position, block layout, objective, evaluation setting을 같이 말해야 합니다.

## 다른 architecture와의 관계

Transformer가 recurrence나 convolution을 “불필요한 개념”으로 만든 것은 아닙니다. 오히려 각 구조의 inductive bias가 무엇인지 더 분명하게 만들었습니다.

| 구조 | 강한 bias | Transformer와의 차이 |
| --- | --- | --- |
| RNN | compact sequential state | Transformer는 token pair interaction을 직접 계산 |
| CNN | locality와 weight sharing | Transformer는 content-dependent mixing |
| GNN | graph relation 기반 message passing | Transformer는 token 사이 dense message passing처럼 볼 수 있음 |
| SSM/Mamba | 긴 sequence scan 효율성 | dense attention의 $T^2$ 비용 문제를 다시 다룸 |
| MoE | conditional parameter routing | Transformer block과 결합되는 경우가 많음 |

이 비교는 중요합니다. Transformer는 bias가 없는 모델이 아닙니다. token 사이 similarity 기반 mixing, positional encoding에 의존하는 order handling, dense interaction matrix라는 강한 설계 선택을 가집니다.

## 앞으로 논문을 읽을 때 재사용할 질문

Transformer 계열 논문을 읽을 때는 이 질문들을 반복해서 쓰면 됩니다.

| 질문 | 확인할 것 |
| --- | --- |
| token은 무엇인가? | wordpiece, residue, atom, patch, graph node, retrieved chunk, tool event |
| attention pattern은 무엇인가? | full, causal, cross, local, sparse, graph-masked |
| position 정보는 무엇인가? | index, segment, chain, graph distance, 3D relation |
| objective는 무엇인가? | translation, LM, masked modeling, contrastive, preference |
| evidence는 무엇인가? | benchmark, ablation, scaling, efficiency, transfer |
| architecture 말고 뭐가 바뀌었나? | data, parameter count, compute, schedule, tokenizer, metric |

이 논문이 longform으로 좋은 이유도 여기에 있습니다. Transformer 하나를 설명하는 글이 아니라, 이후 architecture paper를 읽는 기본 틀을 만들어 줍니다.

## 이 블로그에서의 위치

이 글은 `papers`가 아니라 `posts`에 두는 것이 더 낫습니다. 이유는 분명합니다.

`papers`에는 논문 하나를 재사용 가능한 영어 카드로 남기는 편이 좋습니다. 그래야 나중에 비교표, concept update, reading queue, related paper graph에서 안정적으로 쓸 수 있습니다.

반면 이 글은 독자에게 설명하는 글입니다. 배경을 풀고, 오해를 정리하고, 어떤 순서로 읽어야 하는지 안내합니다. 이런 글은 한글 `posts`에 두는 것이 블로그 성격과 더 잘 맞습니다.

따라서 이 논문은 두 층으로 남깁니다.

| 위치 | 역할 |
| --- | --- |
| [[papers/architectures/attention-is-all-you-need|compact paper note]] | 영어 paper asset |
| 이 글 | 한글 longform explanation |

앞으로도 같은 원칙을 따르면 좋습니다. 대부분의 논문은 compact paper note로 쌓고, 정말 중요한 anchor paper만 한글 longform post로 승격합니다.

## 결론

`Attention Is All You Need`는 Transformer를 제안한 논문입니다. 하지만 더 정확히 말하면, attention을 중심으로 sequence transduction architecture를 다시 설계한 논문입니다.

기억해야 할 것은 세 가지입니다.

첫째, attention은 recurrence의 보조 장치가 아니라 sequence mixing의 중심 연산이 될 수 있습니다.

둘째, Transformer는 attention 수식 하나가 아니라 attention, FFN, residual, normalization, positional encoding, masking, training recipe가 결합된 구조입니다.

셋째, 논문이 증명한 범위와 이후 역사적 의미를 분리해야 합니다. 이 논문은 현대 LLM의 중요한 출발점이지만, 현대 LLM 전체를 직접 설명하는 논문은 아닙니다.

다음에 읽을 경로는 명확합니다. 먼저 [[concepts/architectures/attention|Attention]]과 [[concepts/architectures/transformer|Transformer]]를 개념으로 고정하고, 그 다음 decoder-only language model, scaling law, instruction tuning, long-context attention, state-space model 계열 논문으로 확장하면 됩니다.

## 다음에 볼 노트

- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/index|Architecture papers]]
- [[papers/llm/index|LLM papers]]
