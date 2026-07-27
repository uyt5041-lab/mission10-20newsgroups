# Experiment Summary

## 1. 문제 정의

20 Newsgroups의 18,846개 문서를 20개 주제 중 하나로 분류한다. 단순 Accuracy보다 class별 균형을 반영하는 Macro F1을 중심 지표로 사용한다.

평가 시 test leakage를 막기 위해 전체 데이터의 15%를 official test로 먼저 고정하고, 나머지 85%에서 Stratified 5-fold OOF 평가를 수행한다. Official test는 모델과 설정을 모두 고정한 뒤 마지막에 한 번만 사용한다.

## 2. 초기 가설

초기 모델은 FastText embedding과 BiGRU, Attention을 결합했다. 이 구조는 문맥과 순서를 반영할 수 있지만, 20 Newsgroups에서는 다음 문제가 관찰됐다.

- encoded payload와 반복 token 같은 noisy text
- 긴 문서 truncation
- `alt.atheism`, `soc.religion.christian`, `talk.religion.misc`의 높은 의미 중첩
- 정확한 topic phrase에 대한 sparse lexical signal의 중요성
- 단일 validation split이 주는 불안정성

## 3. Stage 1: Baseline

### 접근

```text
FastText
→ Packed 2-layer Bidirectional GRU
→ Attention
→ Dropout
→ Linear classifier
```

### 목적

- 기본 neural text classification pipeline 구축
- tokenization, vocabulary, padding, packed sequence, Attention 구현 검증
- 이후 실험의 비교 기준 생성

## 4. Stage 2: 학습 안정화

### 문제

- epoch에 따라 validation 성능 변동
- 장기 학습 시 과적합 가능성
- GRU의 exploding gradient 위험

### 해결 접근

- Validation Macro F1 기반 early stopping
- `ReduceLROnPlateau`
- gradient clipping
- gradient accumulation
- fold별 best checkpoint 복원

이 단계에서 모델 구조보다 training recipe와 evaluation protocol을 먼저 고정했다.

## 5. Stage 3: 전처리와 sequence length

### 문제

일부 문서에 uuencode, base64 형태의 encoded noise와 반복 short token이 포함되어 있었다. 또한 `MAX_LEN=300`이 긴 문서의 핵심 문맥을 지나치게 잘라낼 가능성이 있었다.

### 실험

| Experiment | Preprocessing | MAX_LEN |
|---|---|---:|
| F-local | 기존 전처리 | 300 |
| J | 기존 전처리 | 512 |
| K | encoded noise 제거 | 300 |
| L | encoded noise 제거 | 512 |

### 판단 원칙

- 길이 증가가 좋아지면 truncation이 병목
- clean preprocessing이 좋아지면 encoded noise가 병목
- 둘 다 좋아지면 L을 후속 기준으로 사용

## 6. Stage 4: Sparse feature와 OOF 비교

### 문제

Neural model이 문맥은 보지만 정확한 class-specific phrase를 충분히 활용하지 못할 수 있었다.

### 해결 접근

- word TF-IDF
- character TF-IDF
- coarse auxiliary head
- neural/sparse ensemble
- Stratified 5-fold OOF

### 결과

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF | 0.7523 | 0.7424 |
| Ensemble | 0.7454 | 0.7373 |
| Coarse | 0.7306 | 0.7224 |
| Baseline | 0.7316 | 0.7211 |

TF-IDF가 neural model과 ensemble을 모두 앞섰다. 이 데이터에서는 topic phrase와 character n-gram이 강력한 신호임을 확인했다.

## 7. Stage 5: Nested CV와 specialist 검증

### 문제

전체 confusion matrix에서 일부 class group, 특히 religion 내부의 경계가 약했다. Global classifier 뒤에 confusion-group specialist를 붙이면 개선될 수 있다는 가설을 세웠다.

### 해결 접근

- Global LinearSVC
- nested CV
- confusion-group specialist
- 동일 OOF 기준 비교

### OOF 결과

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Nested Global | 0.7598 | 0.7518 |
| Nested Specialist | 0.7595 | 0.7516 |
| Stage 4 fixed | 0.7544 | 0.7449 |

Specialist는 global model보다 좋아지지 않았다. 추가 복잡성이 일반화 이득으로 이어지지 않아 채택하지 않았다.

## 8. Stage 5 official test

최종 Stage 5 global LinearSVC를 고정한 뒤 official test를 한 번 평가했다.

| Metric | OOF | Official test |
|---|---:|---:|
| Accuracy | 0.7598 | 0.7708 |
| Macro F1 | 0.7518 | 0.7659 |
| `talk.religion.misc` F1 | 0.4362 | 0.5732 |

OOF보다 test 성능이 소폭 높아 과도한 CV overfitting 징후는 없었다.

## 9. Stage 6: Subject metadata와 semantic hierarchy

### 새로 정의한 문제

Stage 5 이후 가장 큰 병목은 두 갈래였다.

1. Body-only text에서는 문서 주제를 직접 요약하는 `Subject` 정보가 사라져 있었다.
2. Religion 내부 class는 lexical overlap이 강해 단일 global decision boundary만으로 구분하기 어려웠다.

### 해결 접근

Stage 6에서는 다음 모델을 같은 16,019개 OOF 문서에서 비교했다.

- Body-only LinearSVC
- Subject+Body LinearSVC
- Frozen ModernBERT embedding + LinearSVC
- Global ModernBERT full fine-tuning
- Hierarchical religion head
- Multi-prototype `talk.religion.misc`
- Contrastive loss
- Sparse-semantic score fusion

### 결과

| Model | Accuracy | Macro F1 | misc Precision | misc Recall | misc F1 |
|---|---:|---:|---:|---:|---:|
| Subject+Body LinearSVC | 0.9031 | **0.8999** | 0.8487 | 0.7247 | **0.7818** |
| Global ModernBERT | 0.8908 | 0.8876 | 0.7237 | 0.7603 | 0.7416 |
| Sparse-semantic fusion | 0.8659 | 0.8633 | 0.7204 | **0.7865** | 0.7520 |
| Hierarchical multi-prototype | 0.8614 | 0.8588 | 0.7158 | 0.7828 | 0.7478 |
| Body-only LinearSVC | 0.7643 | 0.7559 | 0.5659 | 0.3539 | 0.4355 |
| Frozen ModernBERT + LinearSVC | 0.7254 | 0.6992 | 0.6400 | 0.0599 | 0.1096 |

### 해석

#### 1. 가장 큰 개선은 model architecture가 아니라 input restoration

Body-only LinearSVC의 Macro F1은 0.7559였지만, Subject를 복원하자 0.8999로 상승했다. 따라서 이전 단계의 핵심 병목은 semantic capacity 부족만이 아니라 중요한 metadata 제거였다.

#### 2. Full fine-tuning은 frozen embedding보다 크게 우수

Frozen ModernBERT + LinearSVC는 Macro F1 0.6992에 그쳤지만, end-to-end fine-tuning한 Global ModernBERT는 0.8876을 기록했다. Pretrained representation을 고정해 쓰는 것과 task-specific fine-tuning은 전혀 다른 결과를 냈다.

#### 3. Hierarchy와 fusion은 recall을 높였지만 전체 균형을 잃음

Sparse-semantic fusion은 `talk.religion.misc` Recall 0.7865로 가장 높았다. Hierarchical multi-prototype도 0.7828을 기록했다. 그러나 precision 하락과 다른 class의 성능 손실로 전체 Macro F1은 Subject+Body LinearSVC보다 낮았다.

#### 4. 복잡한 구조가 항상 우승하지는 않음

Religion-specific inductive bias는 목표 class Recall 개선에는 기여했지만, 최종 objective인 20-class Macro F1에서는 단순한 Subject+Body LinearSVC가 가장 강하고 안정적이었다.

## 10. 최종 결론

Stage 1부터 Stage 6까지의 흐름은 다음과 같다.

```text
Neural baseline 구축
→ training 안정화
→ noisy text와 truncation 점검
→ sparse lexical signal 검증
→ nested specialist 기각
→ Subject metadata 복원
→ ModernBERT와 hierarchical model 비교
```

최종 OOF 기준 우승 모델은 **Subject+Body LinearSVC**다.

- Accuracy: 0.9031
- Macro F1: 0.8999
- `talk.religion.misc` F1: 0.7818

이 결과는 20 Newsgroups에서 강한 pretrained Transformer보다도, 올바른 입력 정보와 적합한 sparse representation이 더 중요한 경우가 있음을 보여준다.
