# Mission 10: 20 Newsgroups Text Classification

20 Newsgroups의 18,846개 문서를 20개 주제로 분류하는 실험 기록입니다.

이 저장소는 단일 모델의 점수만 보여주기보다, **오류를 발견하고 가설을 세운 뒤 다음 실험으로 이어간 과정**을 단계별 notebook과 보고서로 정리합니다.

## Problem

- Task: 20-class text classification
- Dataset: `sklearn.datasets.fetch_20newsgroups(subset="all")`
- Primary metric: Macro F1
- Evaluation policy:
  - 전체 데이터의 15%를 official test로 먼저 고정
  - 나머지 85%에서 Stratified 5-fold OOF 평가
  - 모델 선택 과정에서는 official test를 열지 않음
  - 최종 설정을 고정한 뒤 official test를 한 번만 평가

## Experiment roadmap

| Stage | Main question | Main approach | Key result |
|---|---|---|---|
| 1 | 기본 neural text classifier가 어느 정도 작동하는가? | FastText + BiGRU + Attention | baseline 구축 |
| 2 | 학습 불안정과 과적합을 줄일 수 있는가? | scheduler, early stopping, clipping, accumulation | training recipe 정리 |
| 3 | encoded noise와 truncation이 성능을 막는가? | noise cleaning, `MAX_LEN` 300/512 비교 | 전처리 가설 검증 |
| 4 | sparse lexical signal이 neural model을 보완하는가? | TF-IDF, coarse head, ensemble, 5-fold OOF | TF-IDF Macro F1 0.7424 |
| 5 | 혼동 class specialist가 global model보다 나은가? | nested CV, global LinearSVC, specialist | global OOF 0.7518, test 0.7659 |
| 6 | Subject metadata와 semantic/hierarchical model이 병목을 해결하는가? | Subject+Body LinearSVC, ModernBERT, hierarchical multi-prototype, fusion | Subject+Body LinearSVC OOF Macro F1 0.8999 |

## Repository structure

```text
notebooks/
  01_02_baseline_and_training.ipynb
  03_preprocessing_and_length.ipynb
  04_oof_tfidf_ensemble.ipynb
  05_nested_cv_linearsvc.ipynb
  06_subject_modernbert_hierarchy.ipynb
reports/
  experiment_summary.md
  stage5_test_analysis.md
  stage6_analysis.md
results/
  stage4_summary.md
  stage5_summary.md
  stage6_summary.md
.gitignore
requirements.txt
```

## Main findings

1. 단순히 neural network를 깊게 만드는 것보다 evaluation protocol과 noisy text 처리가 먼저 중요했습니다.
2. 20 Newsgroups에서는 word/character n-gram 기반 sparse feature가 강했습니다.
3. Stage 4에서 TF-IDF가 neural baseline과 ensemble을 앞섰습니다.
4. Stage 5에서는 confusion-group specialist가 global LinearSVC보다 안정적인 이득을 주지 못했습니다.
5. Stage 5 official test에서 global LinearSVC는 Macro F1 0.7659를 기록했고 OOF보다 소폭 높아 일반화가 유지됐습니다.
6. Stage 6에서 `Subject + Body` 입력이 가장 큰 개선을 만들었습니다. Subject+Body LinearSVC는 OOF Macro F1 0.8999, `talk.religion.misc` F1 0.7818을 기록했습니다.
7. Global ModernBERT는 Macro F1 0.8876으로 강했지만 Subject+Body LinearSVC보다 낮았습니다.
8. Hierarchical multi-prototype와 sparse-semantic fusion은 `talk.religion.misc` Recall을 0.78 수준까지 높였지만 전체 Macro F1에서는 단순 Subject+Body LinearSVC를 넘지 못했습니다.

## Stage 6 model comparison

| Model | Accuracy | Macro F1 | misc Precision | misc Recall | misc F1 |
|---|---:|---:|---:|---:|---:|
| Subject+Body LinearSVC | 0.9031 | **0.8999** | 0.8487 | 0.7247 | **0.7818** |
| Global ModernBERT | 0.8908 | 0.8876 | 0.7237 | 0.7603 | 0.7416 |
| Sparse-semantic fusion | 0.8659 | 0.8633 | 0.7204 | **0.7865** | 0.7520 |
| Hierarchical multi-prototype | 0.8614 | 0.8588 | 0.7158 | 0.7828 | 0.7478 |
| Body-only LinearSVC | 0.7643 | 0.7559 | 0.5659 | 0.3539 | 0.4355 |
| Frozen ModernBERT + LinearSVC | 0.7254 | 0.6992 | 0.6400 | 0.0599 | 0.1096 |

## Reproducibility and logs

- 평가 가능한 notebook을 위해 **최종 metric, fold별 결과, 핵심 warning과 runtime 기록은 유지**합니다.
- 반대로 수천 줄의 반복 `tqdm` 출력, debugger warning 반복, model download progress는 가독성을 위해 축약할 수 있습니다.
- model weights, embeddings, OOF arrays는 용량 때문에 저장소에 포함하지 않습니다.
- Stage 6 Kaggle run: https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook
