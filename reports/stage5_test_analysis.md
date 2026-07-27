# Stage 5 Official Test Analysis

## Final model

```text
word TF-IDF (1–2 grams)
+ character TF-IDF (3–5 grams)
→ LinearSVC
```

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7708 |
| Macro Precision | 0.7786 |
| Macro Recall | 0.7623 |
| Macro F1 | 0.7659 |
| `talk.religion.misc` Precision | 0.7143 |
| `talk.religion.misc` Recall | 0.4787 |
| `talk.religion.misc` F1 | 0.5732 |

## Interpretation

OOF Macro F1 0.7518보다 official test Macro F1 0.7659가 소폭 높았다. 따라서 Stage 5 설정이 CV에만 과도하게 맞았다는 증거는 없으며, unseen test에서도 성능이 안정적으로 유지됐다.

`talk.religion.misc`는 precision에 비해 recall이 낮았다. 모델이 misc라고 예측할 때는 비교적 정확하지만, 실제 misc 문서 상당수를 다른 religion class로 보냈다.

주요 confusion:

```text
talk.religion.misc → soc.religion.christian
talk.religion.misc → alt.atheism
```

이는 religion 여부를 판단하는 것보다 religion 내부의 세부 경계를 찾는 문제가 더 어렵다는 뜻이다.

## Decision

- Stage 5 최종 모델: Global LinearSVC
- Specialist: 기각
- Official test 재평가: 하지 않음
- 다음 실험: Subject metadata 및 semantic Transformer 기반 계층 분류
