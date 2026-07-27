# Stage 4 Summary

## Fixed data condition

- clean preprocessing
- `MAX_LEN=300`
- Stratified 5-fold OOF

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF | 0.7523 | 0.7424 |
| Ensemble | 0.7454 | 0.7373 |
| Coarse auxiliary | 0.7306 | 0.7224 |
| BiGRU baseline | 0.7316 | 0.7211 |

## Decision

TF-IDF를 Stage 4 우승 모델로 선택했다. 정확한 word/character n-gram이 20 Newsgroups의 class-specific topic phrase를 효과적으로 포착했다.
