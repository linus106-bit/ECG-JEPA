# CAPTURE-24 경쟁 모델 정리 (Activities of Daily Living)

작성일: 2026-03-18 (UTC)

## 목적

Capture-24 데이터셋의 **"Classifying activities of daily living"** 벤치마크를 학습 전에 빠르게 파악할 수 있도록,
**CAPTURE-24 논문을 인용한 후속 논문들 중** 실제로 모델을 만들고 **Capture-24의 일상생활 활동 분류 벤치마크 결과를 공개한 경우**만 추려 정리했다.

## 포함 기준

- CAPTURE-24 원논문을 인용한 후속 논문일 것.
- 단순 인용이 아니라 실제 모델/학습 파이프라인을 제안하거나 비교 실험을 수행했을 것.
- 결과가 **Capture-24의 10-class daily living activity classification**에 해당한다고 본문/표에서 확인 가능할 것.
- 서로 다른 버전(예: under review 버전과 정식 출판 버전)이 사실상 같은 연구인 경우에는 **정식 출판본을 우선** 사용했다.

## 조사 결과 요약

현재 공개적으로 확인 가능한 자료 중, 위 기준을 **명확하게 만족하면서 Capture-24 ADL 벤치마크 수치를 표로 공개한 후속 논문은 1편**만 확인했다.

- **SensorLLM: Large Language Models Are Super Sensors for Time Series Classification and Beyond** (EMNLP 2025)
  - 링크: https://aclanthology.org/2025.emnlp-main.19/
  - CAPTURE-24를 10개 클래스 분류 문제로 사용했고, 본문 Table 2에 각 모델의 **F1-macro / Accuracy**를 같이 공개했다.
  - 이 논문은 CAPTURE-24를 다음과 같이 사용했다: **151명 중 앞 100명을 train, 나머지 51명을 test**, 샘플링 레이트는 **100Hz → 50Hz 다운샘플링**, HAR 단계에서는 **window size 500 / stride 250**, **10 activity class labels**를 사용.

## Top 10 모델 랭킹 (Capture-24 ADL benchmark)

> 정렬 기준: **F1-macro 내림차순**.  
> 동률은 논문 Table 2에 나온 순서를 유지했다.  
> 출처가 1편뿐이므로, 아래 랭킹은 현재 확인된 **후속 논문 기준 leaderboard-style 정리**로 보면 된다.

| Rank | Model | Paper | F1-macro (%) | Accuracy (%) | 비고 |
|---|---|---|---:|---:|---|
| 1 | SensorLLM | SensorLLM (EMNLP 2025) | 48.6 ± 1.14 | 72.0 ± 0.71 | 논문 제안 모델 |
| 2 | Attend | SensorLLM (EMNLP 2025) | 43.6 ± 0.55 | 71.0 ± 0.71 | HAR baseline |
| 3 | DeepConvLSTMAtt | SensorLLM (EMNLP 2025) | 41.4 ± 0.55 | 70.4 ± 0.55 | HAR baseline |
| 4 | DeepConvLSTM | SensorLLM (EMNLP 2025) | 40.4 ± 0.89 | 69.4 ± 1.14 | HAR baseline |
| 5 | Chronos+MLP | SensorLLM (EMNLP 2025) | 38.0 ± 0.71 | 68.2 ± 0.84 | Chronos 임베딩 + MLP |
| 6 | PatchTST | SensorLLM (EMNLP 2025) | 35.6 ± 0.89 | 66.2 ± 1.10 | TS baseline |
| 7 | Informer | SensorLLM (EMNLP 2025) | 35.6 ± 0.55 | 66.8 ± 0.84 | TS baseline |
| 8 | NS-Transformer | SensorLLM (EMNLP 2025) | 34.8 ± 1.10 | 65.4 ± 0.55 | TS baseline |
| 9 | TimesNet | SensorLLM (EMNLP 2025) | 34.8 ± 0.84 | 65.8 ± 1.79 | TS baseline |
| 10 | Transformer | SensorLLM (EMNLP 2025) | 32.8 ± 0.84 | 65.4 ± 0.89 | TS baseline |

### 컷오프 밖 모델

- GPT4TS — 32.8 ± 1.10 F1-macro, 62.2 ± 1.92 accuracy
- iTransformer — 19.8 ± 0.84 F1-macro, 62.4 ± 0.89 accuracy

## 해석 메모

1. **현재 확인된 후속 논문 기준 최고 성능은 SensorLLM의 48.6 macro-F1**이다.
2. 전통적인 HAR 계열 모델 중에서는 **Attend (43.6)**, **DeepConvLSTMAtt (41.4)**, **DeepConvLSTM (40.4)** 순으로 강했다.
3. 순수 시계열 Transformer 계열은 CAPTURE-24 같은 **free-living, class imbalance가 큰 10-class 문제**에서 HAR 전용 모델보다 약한 편이었다.
4. 따라서 네가 Capture-24 ADL 벤치마크에서 경쟁력을 보려면, 최소한 **Attend/DeepConvLSTMAtt 급**은 넘는 것을 1차 목표로 보고,
   **48.6 macro-F1**를 현재 확인된 공개 후속논문 기준 SOTA 후보선으로 보면 된다.

## 참고: 원 논문 내부 benchmark와의 관계

이 문서는 **후속 인용 논문**만 모은 것이다.  
다만 비교 감각을 위해 원 CAPTURE-24 논문을 보면, "Classifying activities of daily living"에서
**CNN + HMM = 0.576 F1-score**가 내부 benchmark 최고였다.

주의할 점은, 이 값은 원 논문의 자체 실험 설정에서 나온 값이고,
위 SensorLLM 표는 **다른 전처리/윈도우/학습 설정**(예: 50Hz downsampling, window size 500, 5 random runs)을 사용했기 때문에
**숫자를 일대일로 직접 비교하면 안 된다.**

## 사용한 주요 원문 소스

1. CAPTURE-24 원논문
   - Chan et al., *CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition*.
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11484779/
2. 후속 논문(정식 출판본)
   - Liu et al., *SensorLLM: Large Language Models Are Super Sensors for Time Series Classification and Beyond*.
   - ACL Anthology: https://aclanthology.org/2025.emnlp-main.19/

## 추가 메모

추가로 찾은 몇몇 CAPTURE-24 관련 후속 연구들은
- CAPTURE-24를 **사전학습(source dataset)** 으로만 사용하거나,
- Capture-24 결과를 **ADL 10-class가 아닌 다른 태스크/설정**으로 제시하거나,
- 본문에서 일상생활 분류 수치를 명확히 표로 공개하지 않아,
이번 정리에서는 제외했다.
