# Notebooks

단계별 원본 실행 notebook과 저장소 내 정리 경로입니다. 대용량 output이 포함된 원본 notebook은 Google Drive, Kaggle, Colab에 보관하고 GitHub에서는 아래 경로 체계로 안내합니다.

| Stage | GitHub 정리 경로 | 원본 실행 notebook | 핵심 내용 |
|---|---|---|---|
| 1–2 | `notebooks/01_02_baseline_and_training.ipynb` | [미션10_10_팀명_성함.ipynb](https://drive.google.com/file/d/1PW0kbX486_qQKHGXJoQ0oylZrhHgKpuV/view?usp=drivesdk) | FastText + BiGRU + Attention baseline, 학습 안정화 |
| 3 | `notebooks/03_preprocessing_and_length.ipynb` | [mission10_stage3_mps_local.ipynb](https://drive.google.com/file/d/1vbe8sR6wkm1-7_OCIFa9sbI5_WENGD2E/view?usp=drivesdk) | encoded noise 제거, sequence length 비교 |
| 4 | `notebooks/04_oof_tfidf_ensemble.ipynb` | [mission10_stage4_kaggle_gpu.ipynb](https://drive.google.com/file/d/1qfTxisT4aMFhLSPbIRfnjYXEJsWIDAZh/view?usp=drivesdk) | 5-fold OOF, TF-IDF, coarse head, ensemble |
| 5 | `notebooks/05_nested_cv_linearsvc.ipynb` | [mission10_stage5_nested_cv_kaggle_results.ipynb](https://drive.google.com/file/d/1jiqlfABdl1Ap1csf-GdUeIuL27qftDwF/view?usp=drivesdk) | Nested CV, global LinearSVC, specialist 검증 |
| 6 | `notebooks/06_subject_modernbert_hierarchy.ipynb` | [Stage 6 Kaggle notebook](https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook) | Subject+Body LinearSVC, ModernBERT, hierarchy, fusion |
| Official test | `notebooks/07_stage6_official_test_once.ipynb` | [Stage 6 Official Test Colab](https://colab.research.google.com/drive/1Igk8Bjw43qUYsRXVitCR6Yv0BcSAFyd7) | 최종 모델 고정 후 holdout 2,827개 1회 평가 |

## 결과 문서 경로

| 구분 | GitHub 경로 |
|---|---|
| 전체 실험 요약 | [`reports/experiment_summary.md`](../reports/experiment_summary.md) |
| Stage 5 test 분석 | [`reports/stage5_test_analysis.md`](../reports/stage5_test_analysis.md) |
| Stage 6 결과 요약 | [`results/stage6_summary.md`](../results/stage6_summary.md) |

## 제출용 notebook

제출 파일은 Stage 1–2 메인 notebook을 기반으로 하되, 맨 앞에 전체 실험 요약과 최종 결과를 배치하고 GitHub, Kaggle, Colab 링크를 함께 포함합니다.
