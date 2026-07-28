# 결과 요약 / 목차 - Mission 10: 20 Newsgroups Text Classification

## 1. 최종 결과

20 Newsgroups의 18,846개 문서를 20개 주제로 분류하는 실험을 진행했다.

최종 모델은 **Subject + Body TF-IDF + LinearSVC**이며,  
고정된 official test 2,827개 문서에서 다음 성능을 기록했다.

| Metric | OOF | Official Test |
|---|---:|---:|
| Accuracy | 0.9031 | **0.9066** |
| Macro F1 | 0.8999 | **0.9046** |
| `talk.religion.misc` Precision | 0.8487 | **0.9359** |
| `talk.religion.misc` Recall | 0.7247 | **0.7766** |
| `talk.religion.misc` F1 | 0.7818 | **0.8488** |

최종 모델은 OOF보다 official test에서 성능이 소폭 상승해,  
교차검증 결과가 실제 holdout test에서도 안정적으로 일반화됨을 확인했다.

특히 주요 병목 클래스였던 `talk.religion.misc`의 F1이  
Stage 5 official test의 0.5732에서 Stage 6 official test의 0.8488로 크게 개선됐다.

---

## 2. 목차: 실험 흐름 및 Notebook 구성

### A. 본 제출 Notebook에 포함된 실험

| 구분 | 실험 내용 | 실행 Notebook |
|---|---|---|
| Baseline | Word2Vec·FastText·GloVe + BiGRU 비교 | [`mission10_baseline_to_stage2_colab.ipynb`](notebooks/mission10_baseline_to_stage2_colab.ipynb) |
| Stage 1 (1차 개선) | 학습 설정 및 모델 개선 ablation | 위 제출 notebook에 포함 |
| Stage 2 (2차 개선) | Packed Attention BiGRU 및 추가 ablation | 위 제출 notebook에 포함 |

### B. 별도 Notebook에서 수행한 후속 실험

| Stage | 실험 내용 | 실행 Notebook |
|---|---|---|
| Stage 3 (3차 개선) | Encoded noise 제거 및 sequence length 비교 | [`mission10_stage3_mps_local.ipynb`](notebooks/mission10_stage3_mps_local.ipynb) |
| Stage 4 | TF-IDF, coarse head, ensemble, 5-fold OOF | [`mission10_stage4_kaggle_gpu.ipynb`](notebooks/mission10_stage4_kaggle_gpu.ipynb) |
| Stage 5 | Nested CV, Global LinearSVC, specialist 검증 및 Stage 5 official test | [`mission10_stage5_final_test_colab.ipynb`](notebooks/mission10_stage5_final_test_colab.ipynb) |
| Stage 6 | Subject+Body, ModernBERT, hierarchy, fusion 비교 | [Stage 6 Kaggle Notebook](https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook) |
| Final Test | 최종 모델의 official test 1회 평가 | [`mission10_stage6_final_test_colab.ipynb`](notebooks/mission10_stage6_final_test_colab.ipynb) |

Baseline과 Stage 1·2는 제출 notebook에서 확인할 수 있으며, Stage 3부터 Final Test까지는 위 GitHub 및 Kaggle 경로에서 확인할 수 있다.

---

## 3. 전체 실험 결과 요약

| 구분 | 핵심 실험 | 최고 결과 | 주요 결론 |
|---|---|---:|---|
| Baseline | Word2Vec·FastText·GloVe + BiGRU | Macro F1 0.7211 | FastText 기반 baseline 채택 |
| 1차 개선 | 학습 안정화 및 구조 ablation | baseline 대비 비교 | 학습 안정화 조건 검증 |
| 2차 개선 | Packed Attention BiGRU 개선 | 추가 비교 실험 | 복잡도 증가의 실효성 점검 |
| 3차 개선 | Noise cleaning 및 sequence length | K(clean, 300) Val F1 0.7221 / Test F1 0.7170 | 별도 MPS notebook에서 실행 완료 |
| Stage 4 | TF-IDF 및 5-fold OOF | 0.7424 | Sparse lexical feature가 neural model을 상회 |
| Stage 5 | Nested Global LinearSVC | 0.7518 | Specialist보다 global model이 안정적 |
| Stage 5 Test | Official test | 0.7659 | OOF 성능이 holdout에서도 유지 |
| Stage 6 | Subject+Body LinearSVC | 0.8999 | Subject 복원이 가장 큰 개선 |
| Final Test | Subject+Body LinearSVC | **0.9046** | 최종 일반화 성능 확인 |

---

## 4. 최종 결론

이번 프로젝트 실험에서 가장 큰 성능 향상은 복잡한 모델 구조가 아니라  
제거되어 있던 **Subject metadata를 복원한 것**에서 발생했다.

초기 실험에서는 작성자, 이메일, 조직명 등 header 기반 leakage를 줄이기 위해 headers를 제거했다. 그러나 이 과정에서 문서의 주제를 직접 요약하는 Subject 정보도 함께 제거되었다. Stage 6에서는 raw header에서 Subject만 선택적으로 복원하고 나머지 header 정보는 사용하지 않았다.

Body-only LinearSVC의 Macro F1은 0.7559였지만,  
Subject와 Body를 함께 사용하자 0.8999까지 상승했다.

ModernBERT, hierarchical classifier, multi-prototype, fusion 모델도 비교했지만,  
전체 Macro F1 기준으로는 단순하고 해석 가능한  
**Subject + Body TF-IDF + LinearSVC**가 가장 높은 성능을 기록했다.

따라서 이 데이터셋에서는 모델 복잡도보다  
적절한 입력 정보, sparse lexical representation, 안정적인 평가 설계가 더 중요했다.

#### 제목을 포함하여 학습 후 결과가 좋아진 점에 대하여

초기 baseline에도 header에 있던 Subject를 넣었다면 점수는 상당히 높아졌을 가능성이 크다. 다만 최종 0.90은 Subject 복원과 TF-IDF·char n-gram·LinearSVC 조합이 함께 만든 결과라서, neural baseline도 동일한 점수에 도달했을 것이라고 단정할 수는 없다.

결론적으로, 데이터의 성격과 구조를 살피는 것이 중요했다.

---

## 5. 전체 코드 및 실행 결과

- GitHub  
  https://github.com/uyt5041-lab/mission10-20newsgroups

- Kaggle Stage 6 전체 실험  
  https://www.kaggle.com/code/chattybeak/mission-10-stage-6/notebook

- Stage 6 Official Test Colab  
  https://colab.research.google.com/drive/1Igk8Bjw43qUYsRXVitCR6Yv0BcSAFyd7
