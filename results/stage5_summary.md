# Stage 5 Summary

## Nested OOF results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Nested Global LinearSVC | 0.7598 | 0.7518 |
| Nested Specialist | 0.7595 | 0.7516 |
| Stage 4 fixed SGD | 0.7544 | 0.7449 |

## Specialist contribution

- Specialist minus Global: -0.0003 Macro F1
- Specialist minus Stage 4 fixed: +0.0066 Macro F1

The specialist did not outperform the global classifier, so it was rejected.

## Official test

| Metric | Value |
|---|---:|
| Accuracy | 0.7708 |
| Macro F1 | 0.7659 |
| `talk.religion.misc` F1 | 0.5732 |

The official test was evaluated once after the final Stage 5 configuration was fixed.
