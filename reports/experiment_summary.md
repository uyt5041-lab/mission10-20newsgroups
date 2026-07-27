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

## 8. Official test

최종 Stage 5 global LinearSVC를 고정한 뒤 official test를 한 번 평가했다.

| Metric | OOF | Official test |
|---|---:|---:|
| Accuracy | 0.7598 | 0.7708 |
| Macro F1 | 0.7518 | 0.7659 |
| `talk.religion.misc` F1 | 0.4362 | 0.5732 |

OOF보다 test 성능이 소폭 높아 과도한 CV overfitting 징후는 없었다.

## 9. 남은 병목

- `talk.religion.misc` Recall 부족
- `soc.religion.christian`, `alt.atheism`, `talk.religion.misc` 간 의미 중첩
- `rec.autos`의 과다 예측
- Subject metadata 부재 시 성능 저하

이 분석이 Stage 6의 Subject+Body, ModernBERT, hierarchical religion head, multi-prototype misc, contrastive loss 실험으로 이어졌다.
