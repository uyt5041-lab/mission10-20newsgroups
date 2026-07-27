# Mission 10 Stage 6 Result

## OOF comparison

| Model | Accuracy | Macro F1 | misc Precision | misc Recall | misc F1 |
|---|---:|---:|---:|---:|---:|
| Subject+Body LinearSVC | 0.9031 | **0.8999** | 0.8487 | 0.7247 | **0.7818** |
| Global ModernBERT | 0.8908 | 0.8876 | 0.7237 | 0.7603 | 0.7416 |
| Sparse-semantic fusion | 0.8659 | 0.8633 | 0.7204 | **0.7865** | 0.7520 |
| Hierarchical multi-prototype | 0.8614 | 0.8588 | 0.7158 | 0.7828 | 0.7478 |
| Body-only LinearSVC | 0.7643 | 0.7559 | 0.5659 | 0.3539 | 0.4355 |
| Frozen ModernBERT + LinearSVC | 0.7254 | 0.6992 | 0.6400 | 0.0599 | 0.1096 |

## Selection

- OOF winner: **Subject+Body LinearSVC**
- OOF Macro F1: **0.8999**
- OOF `talk.religion.misc` F1: **0.7818**
- Target F1 0.60 achieved: **Yes**

## Official test

The selected configuration was frozen and evaluated once on the 2,827-document holdout set.

| Metric | OOF | Official test |
|---|---:|---:|
| Accuracy | 0.9031 | **0.9066** |
| Macro F1 | 0.8999 | **0.9046** |
| `talk.religion.misc` Precision | 0.8487 | **0.9359** |
| `talk.religion.misc` Recall | 0.7247 | **0.7766** |
| `talk.religion.misc` F1 | 0.7818 | **0.8488** |

## Interpretation

- Subject restoration produced the largest gain.
- Full ModernBERT fine-tuning was much stronger than frozen embeddings, but did not beat the sparse Subject+Body model.
- Hierarchical and fusion models increased `talk.religion.misc` Recall, but lost overall Macro F1.
- The simplest high-performing model remained the final OOF winner.
- Official test performance slightly exceeded OOF performance, with no sign of severe CV overfitting.
- The holdout result was sealed after one evaluation and was not reused for further tuning.

## Source

- Kaggle run: https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook
- Official test Colab: https://colab.research.google.com/drive/1Igk8Bjw43qUYsRXVitCR6Yv0BcSAFyd7
- Drive artifact: `stage6_model_comparison.csv`
