# Notebooks

아래는 단계별 원본 notebook입니다. GitHub 저장소에는 보고서와 결과 요약을 먼저 정리했으며, 원본 실행 notebook은 Google Drive 링크로 보관합니다.

| Stage | Notebook | Description |
|---|---|---|
| 1–2 | [미션10_10_팀명_성함.ipynb](https://drive.google.com/file/d/1PW0kbX486_qQKHGXJoQ0oylZrhHgKpuV/view?usp=drivesdk) | FastText + BiGRU + Attention baseline, training 개선 |
| 3 | [mission10_stage3_mps_local.ipynb](https://drive.google.com/file/d/1vbe8sR6wkm1-7_OCIFa9sbI5_WENGD2E/view?usp=drivesdk) | encoded noise, sequence length 실험 |
| 4 | [mission10_stage4_kaggle_gpu.ipynb](https://drive.google.com/file/d/1qfTxisT4aMFhLSPbIRfnjYXEJsWIDAZh/view?usp=drivesdk) | 5-fold OOF, TF-IDF, coarse head, ensemble |
| 5 | [mission10_stage5_nested_cv_kaggle_results.ipynb](https://drive.google.com/file/d/1jiqlfABdl1Ap1csf-GdUeIuL27qftDwF/view?usp=drivesdk) | Nested CV, LinearSVC, specialist 검증 |

## Naming plan

원본 notebook을 저장소에 직접 추가할 때는 다음 이름을 사용합니다.

```text
01_02_baseline_and_training.ipynb
03_preprocessing_and_length.ipynb
04_oof_tfidf_ensemble.ipynb
05_nested_cv_linearsvc.ipynb
```
