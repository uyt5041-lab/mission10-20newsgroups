# Mission 10: 20 Newsgroups Text Classification

20 Newsgroups의 18,846개 문서를 20개 주제로 분류하는 실험입니다.

단순히 최고 점수만 제시하지 않고, neural baseline에서 시작해 전처리, sparse feature, nested CV, Subject metadata, ModernBERT와 hierarchical model을 차례로 검증한 과정을 정리했습니다.

## 1. 최종 결과

최종 선택 모델은 **Subject + Body TF-IDF + LinearSVC**입니다.

| Metric | OOF | Official Test |
|---|---:|---:|
| Accuracy | 0.9031 | **0.9066** |
| Macro F1 | 0.8999 | **0.9046** |
| `talk.religion.misc` Precision | 0.8487 | **0.9359** |
| `talk.religion.misc` Recall | 0.7247 | **0.7766** |
| `talk.religion.misc` F1 | 0.7818 | **0.8488** |

Official test는 모델과 설정을 고정한 뒤 2,827개 holdout 문서에 한 번만 수행했습니다. OOF보다 Macro F1이 소폭 높게 유지되어 교차검증 성능이 실제 holdout에서도 안정적으로 일반화됐습니다.

주요 병목 클래스였던 `talk.religion.misc`의 F1은 Stage 5 official test의 0.5732에서 최종 0.8488로 개선됐습니다.

## 2. 문제 및 평가 설계

- Task: 20-class text classification
- Dataset: `sklearn.datasets.fetch_20newsgroups(subset="all")`
- Primary metric: Macro F1
- Seed: 42
- 전체 데이터의 15%를 official test로 먼저 고정
- 나머지 85%에서 Stratified 5-fold OOF 평가
- 모델 선택 과정에서는 official test를 사용하지 않음
- 최종 설정 고정 후 official test를 한 번만 평가

> 초기 single split 실험과 Stage 4 이후의 5-fold OOF는 평가 방식이 다르므로 수치를 완전히 같은 조건으로 비교하지 않습니다.

## 3. 실험 흐름 및 Notebook

### A. 제출용 main notebook에 포함된 실험

| 구분 | 실험 내용 | GitHub Notebook |
|---|---|---|
| Baseline | Word2Vec, FastText, GloVe + BiGRU 비교 | [`mission10_baseline_to_stage2_colab.ipynb`](notebooks/mission10_baseline_to_stage2_colab.ipynb) |
| 1차 개선 | 학습 설정 및 모델 개선 ablation | 위 main notebook에 포함 |
| 2차 개선 | Packed Attention BiGRU 및 추가 ablation | 위 main notebook에 포함 |

### B. 별도 notebook에서 수행한 후속 실험

| 구분 | 실험 내용 | 실행 경로 |
|---|---|---|
| 3차 개선 | Encoded noise 제거 및 sequence length 비교 | [`mission10_stage3_mps_local.ipynb`](notebooks/mission10_stage3_mps_local.ipynb) |
| Stage 4 | TF-IDF, coarse head, ensemble, 5-fold OOF | [`mission10_stage4_kaggle_gpu.ipynb`](notebooks/mission10_stage4_kaggle_gpu.ipynb) |
| Stage 5 | Nested CV, Global LinearSVC, specialist 및 official test | [`mission10_stage5_final_test_colab.ipynb`](notebooks/mission10_stage5_final_test_colab.ipynb) |
| Stage 6 전체 실험 | Subject+Body, ModernBERT, hierarchy, fusion 비교 | [Kaggle Stage 6 run](https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook) |
| Final Test | 최종 Subject+Body LinearSVC official test | [`mission10_stage6_final_test_colab.ipynb`](notebooks/mission10_stage6_final_test_colab.ipynb) |

## 4. 전체 실험 결과 요약

| 구분 | 핵심 실험 | 최고 결과 | 주요 결론 |
|---|---|---:|---|
| Baseline | Word2Vec, FastText, GloVe + BiGRU | Macro F1 0.7211 | FastText 기반 baseline 채택 |
| 1차 개선 | 학습 안정화 및 구조 ablation | baseline 대비 비교 | scheduler, early stopping, clipping 등 검증 |
| 2차 개선 | Packed Attention BiGRU 개선 | 추가 비교 실험 | 복잡도 증가의 실효성 점검 |
| 3차 개선 | Noise cleaning 및 sequence length | K(clean, 300) Val F1 0.7221 / Test F1 0.7170 | 별도 MPS notebook에서 실행 완료 |
| Stage 4 | TF-IDF 및 5-fold OOF | Macro F1 0.7424 | Sparse lexical feature가 neural model을 상회 |
| Stage 5 | Nested Global LinearSVC | OOF 0.7518 / Test 0.7659 | Specialist보다 global model이 안정적 |
| Stage 6 | Subject+Body LinearSVC | OOF **0.8999** | Subject 복원이 가장 큰 개선 |
| Final Test | Subject+Body LinearSVC | Test **0.9046** | 최종 일반화 성능 확인 |

## 5. 가장 큰 개선이 발생한 이유

초기 실험에서는 작성자, 이메일, 조직명 등 header 기반 leakage를 줄이기 위해 다음 설정으로 데이터를 불러왔습니다.

```python
fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
```

그러나 `Subject:`도 header에 포함되어 있어, 이 과정에서 문서 주제를 직접 요약하는 제목까지 함께 제거됐습니다.

Stage 6에서는 raw 문서에서 **Subject만 선택적으로 추출**하고, 작성자·이메일·조직명 등 나머지 header 정보는 사용하지 않은 채 clean Body와 결합했습니다.

```text
Raw document → Subject만 추출
Clean dataset → Body 사용
Subject + Body → 최종 입력
```

같은 sparse classifier 계열에서 다음 차이가 관찰됐습니다.

| Input | Macro F1 |
|---|---:|
| Body-only LinearSVC | 0.7559 |
| Subject+Body LinearSVC | **0.8999** |

따라서 가장 큰 병목은 단순한 model capacity 부족이 아니라 **중요한 입력 정보의 제거**였습니다.

다만 최종 0.90 성능은 Subject 복원만의 결과가 아니라, word TF-IDF, character n-gram, LinearSVC가 함께 만든 결과입니다. 초기 neural baseline에 Subject를 넣었더라도 성능은 상당히 상승했을 가능성이 크지만 동일한 0.90에 도달했다고 단정할 수는 없습니다.

## 6. Stage 6 OOF 모델 비교

| Model | Accuracy | Macro F1 | misc Precision | misc Recall | misc F1 |
|---|---:|---:|---:|---:|---:|
| Subject+Body LinearSVC | 0.9031 | **0.8999** | 0.8487 | 0.7247 | **0.7818** |
| Global ModernBERT | 0.8908 | 0.8876 | 0.7237 | 0.7603 | 0.7416 |
| Sparse-semantic fusion | 0.8659 | 0.8633 | 0.7204 | **0.7865** | 0.7520 |
| Hierarchical multi-prototype | 0.8614 | 0.8588 | 0.7158 | 0.7828 | 0.7478 |
| Body-only LinearSVC | 0.7643 | 0.7559 | 0.5659 | 0.3539 | 0.4355 |
| Frozen ModernBERT + LinearSVC | 0.7254 | 0.6992 | 0.6400 | 0.0599 | 0.1096 |

Global ModernBERT는 강한 성능을 기록했지만 Subject+Body LinearSVC를 넘지 못했습니다. Hierarchical 및 fusion 방식은 `talk.religion.misc` Recall을 높였으나 전체 Macro F1에서는 손실이 발생했습니다.

## 7. Repository 구조

```text
notebooks/
  mission10_baseline_to_stage2_colab.ipynb
  mission10_stage3_mps_local.ipynb
  mission10_stage4_kaggle_gpu.ipynb
  mission10_stage5_final_test_colab.ipynb
  mission10_stage6_final_test_colab.ipynb
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

## 8. 보고서 및 실행 링크

- [전체 실험 보고서](reports/experiment_summary.md)
- [Stage 5 test 분석](reports/stage5_test_analysis.md)
- [Stage 6 결과 요약](results/stage6_summary.md)
- [Stage 6 전체 Kaggle 실행](https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook)
- [Stage 6 Official Test Colab](https://colab.research.google.com/drive/1Igk8Bjw43qUYsRXVitCR6Yv0BcSAFyd7)

## 9. 최종 결론

이번 프로젝트에서 가장 큰 성능 향상은 복잡한 architecture 추가보다 **Subject metadata 복원**에서 발생했습니다.

20 Newsgroups에서는 올바른 입력 정보, word/character n-gram 기반 sparse representation, leakage를 방지한 평가 설계가 모델 복잡도보다 더 중요했습니다.

최종 모델은 단순하고 해석 가능한 **Subject + Body TF-IDF + LinearSVC**이며, official test에서 Accuracy 0.9066, Macro F1 0.9046을 기록했습니다.

Official test 결과를 확인한 뒤 같은 holdout을 이용한 추가 tuning은 수행하지 않았습니다.
