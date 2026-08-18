# Promptune V6 Final Model Selection Policy

## Status

**Benchmark-blind freeze document**

이 문서는 Seed 42/52/62의 V6 validation 결과와 내부 진단을 확인한 이후,
**Final Benchmark 210을 열기 전에** 최종 모델 선택 원칙을 고정하기 위한 문서다.

따라서 이를 preregistration이라고 부르지 않는다.
정확한 표현은 **Final Benchmark 이전 selection policy freeze**다.

---

## 1. 현재 후보

공식 V6 baseline:

- Seed 42
- Seed 52
- Seed 62

모든 공식 run은 동일한:

- Git commit
- Train/Valid SHA256
- Model
- Epoch
- Batch size
- Learning rate
- Max length
- Threshold tuning policy

를 사용해야 한다.

---

## 2. Primary selection criterion

**V1 tuned Macro F1 point estimate**

이 프로젝트는 8개 Promptune 요소를 모두 다루는 multi-label classifier이므로,
빈도가 높은 Context에 전체 판단이 과도하게 끌리지 않도록 Macro F1을 Primary criterion으로 둔다.

---

## 3. Supporting diagnostics

Primary ranking을 해석할 때 다음을 함께 확인한다.

1. Micro F1
2. Exact Match
3. 8개 label별 Precision / Recall / F1
4. Threshold-independent Average Precision
5. Paired bootstrap diagnostic
6. rare-label support와 F1 uncertainty
7. training log / NaN / divergence / data mismatch 여부
8. 특정 핵심 label에서의 비정상적인 collapse 여부

Supporting diagnostic은 Primary criterion을 임의의 가중치 점수로 대체하지 않는다.

---

## 4. Statistical interpretation rule

Paired bootstrap difference interval이 0을 포함하는 경우:

> 해당 Seed가 validation point estimate상 더 높더라도
> 다른 Seed보다 통계적으로 우수하다고 주장하지 않는다.

운영상 하나의 모델을 선택해야 할 경우 Primary criterion의 최고 point estimate를
leading/selected candidate로 사용할 수 있으나, 이는 **operational selection**이지
**statistically significant superiority claim**이 아니다.

---

## 5. Threshold rule

Seed별 최적 threshold의 단순 평균/median을 최종 threshold로 자동 사용하지 않는다.

이유:

- Seed마다 score scale/calibration이 달라질 수 있다.
- 같은 성능 plateau에서도 threshold 위치는 달라질 수 있다.

최종 threshold는:

> 최종 선택 모델의 validation probability distribution을 기준으로 결정

한다.

Threshold stability는 diagnostic으로 사용하되,
performance certainty와 동일한 개념으로 해석하지 않는다.

---

## 6. Rare-label rule

Validation positive support가 매우 작은 label의 단일 F1은 강하게 해석하지 않는다.

특히:

- Length
- Format
- Audience
- Tone
- Example

은 point estimate와 함께 support 및 uncertainty를 보고한다.

Positive support가 극소수인 label에서 bootstrap threshold 범위가 좁게 나오더라도,
가능한 resampling composition 자체가 제한될 수 있으므로 이를 강한 안정성 증거로 사용하지 않는다.

---

## 7. Error-analysis rule

Final Benchmark 이전에 validation FP/FN을 사람 검토할 수 있다.

이 검토는:

- model failure pattern 이해
- annotation ambiguity 확인
- V6.1 필요 여부 결정

을 위한 **development analysis**다.

Validation 오류 분석을 V6.1 설계에 사용했다면 Valid 400을 더 이상
독립적인 최종 일반화 평가셋으로 해석하지 않는다.

---

## 8. Final Benchmark rule

Final Benchmark 210은 다음이 모두 완료된 뒤에만 사용한다.

1. V6 유지 / V6.1 진행 결정
2. 최종 학습 설정 freeze
3. 최종 Seed freeze
4. 최종 threshold policy freeze
5. 최종 model artifact freeze

Benchmark prediction을 확인한 뒤:

- Seed 변경 금지
- Threshold 재튜닝 금지
- Loss 변경 금지
- 데이터 보강 금지
- 모델 재선택 금지

최종 prediction을 고정한 뒤 bootstrap confidence interval을 계산하는 것은 허용한다.
이는 model selection/tuning이 아니라 고정된 test prediction의 uncertainty reporting이다.

---

## 9. 현재 표현 원칙

허용:

> Seed42는 V1 Macro F1 및 Micro F1 point estimate 기준 leading candidate였다.

허용:

> Paired bootstrap diagnostic에서 Seed 간 difference interval이 0을 포함해
> 통계적으로 유의한 Seed 우열은 주장하지 않았다.

금지:

> Seed42가 통계적으로 가장 우수한 모델임을 증명했다.

금지:

> 3개 Seed 실험으로 모델의 전체 일반화 안정성을 검증했다.

---

## 10. 변경 관리

이 문서를 변경해야 하는 경우:

- 변경 이유
- 변경 시점
- Final Benchmark 사용 여부

를 Git commit에 명시한다.

Final Benchmark 결과를 본 뒤 selection rule을 변경하지 않는다.
