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

## Repository structure

```text
notebooks/
  01_02_baseline_and_training.ipynb
  03_preprocessing_and_length.ipynb
  04_oof_tfidf_ensemble.ipynb
  05_nested_cv_linearsvc.ipynb
reports/
  experiment_summary.md
  stage5_test_analysis.md
results/
  stage4_summary.md
  stage5_summary.md
.gitignore
requirements.txt
```

## Main findings

1. 단순히 neural network를 깊게 만드는 것보다 evaluation protocol과 noisy text 처리가 먼저 중요했습니다.
2. 20 Newsgroups에서는 word/character n-gram 기반 sparse feature가 강했습니다.
3. Stage 4에서 TF-IDF가 neural baseline과 ensemble을 앞섰습니다.
4. Stage 5에서는 confusion-group specialist가 global LinearSVC보다 안정적인 이득을 주지 못했습니다.
5. Official test에서 global LinearSVC는 Macro F1 0.7659를 기록했고, OOF보다 소폭 높아 일반화가 유지됐습니다.
6. `talk.religion.misc`는 여전히 `soc.religion.christian`, `alt.atheism`과 강하게 겹쳐 Stage 6의 hierarchical/Transformer 실험으로 이어졌습니다.

## Notes

- 긴 `tqdm` 및 debugger 로그는 GitHub 가독성을 위해 일부 축약했습니다.
- model weights, embeddings, OOF arrays는 용량 때문에 저장소에 포함하지 않습니다.
- Stage 6은 현재 별도 GPU 실험 중이며 결과가 확정되면 추가할 예정입니다.
