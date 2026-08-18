# Promptune V6 Error Analysis Guide

## 목적

V6 3-Seed baseline에서 발생한 FP/FN을 사람 눈으로 검토해,
다음 개선이 실제로 필요한지 결정한다.

후보:

- V6 유지
- hard-negative / 데이터 보강
- weighted BCE
- alternative split sensitivity check
- 기타 최소 실험

---

## 우선 분석

### Rare / difficult labels

1. Audience
2. Tone
3. Length
4. Format
5. Example

### Seed sensitivity diagnostic

- Task: Seed42 vs Seed62 비교

---

## 리뷰 카테고리

각 오류에 대해 하나의 주 카테고리를 선택한다.

### `genuine_model_error`

Ground Truth가 명확하고 모델이 잘못 판단한 사례.

### `label_ambiguity`

라벨 가이드로도 사람 간 판단이 갈릴 가능성이 높은 경계 사례.

### `suspected_annotation_error`

현재 Ground Truth 자체가 잘못되었을 가능성이 있는 사례.

### `lexical_shortcut`

특정 단어·표현에 모델이 과민 반응한 것으로 의심되는 사례.

### `missing_hard_negative`

해당 label의 positive처럼 보이지만 실제로는 negative인 유사 사례가
Train에 충분하지 않을 가능성이 높은 사례.

### `context_reasoning_failure`

단어 자체보다 전체 업무 맥락을 고려해야 하는데 모델이 이를 놓친 사례.

### `other`

위 범주로 설명하기 어려운 사례.

---

## 리뷰 시 질문

1. 이 GT는 라벨링 가이드상 명확한가?
2. 모델이 어떤 단어/구문에 반응했을 가능성이 있는가?
3. 동일 유형의 FP/FN이 여러 Seed에서 반복되는가?
4. 같은 패턴이 Train에 충분히 존재하는가?
5. 모델이 틀린 것인가, label boundary가 애매한 것인가?
6. 새로운 학습 예시가 실제로 문제 해결에 도움될 유형인가?

---

## Hard-negative Leakage Guard

Validation 오류를 보고 데이터를 개선할 때 다음을 지킨다.

### 금지

- Valid 문장을 Train에 그대로 추가
- Valid 문장을 일부 단어만 바꿔 Train에 추가
- Valid 문장을 LLM에 제공하고 "비슷한 문장 생성" 요청
- Valid FP/FN을 사실상 암기하도록 만드는 근접 변형

### 권장

```text
Valid 오류 사례
      ↓
오류 카테고리 추상화
      ↓
개별 Valid 원문은 전달하지 않음
      ↓
다른 팀원이 새로운 상황/표현의 독립 사례 작성
      ↓
사람 라벨링
      ↓
near-duplicate 검토
      ↓
새 training-only data
```

가능하면 오류 분석자와 신규 예시 작성자를 분리한다.

AI 생성 시에도 원문이 아니라 추상적 오류 카테고리만 입력한다.

---

## 중요한 데이터 역할 구분

Valid 400의 오류 패턴을 본 뒤 V6.1을 개선하면,
Valid 400은 실질적으로 development set 역할을 한다.

따라서 최종 일반화 주장은:

> untouched Final Benchmark 210

에서만 한다.

---

## Decision Gate

### Hard-negative / data augmentation 우선 후보

다음이 반복되면 고려:

- 특정 negative 패턴에서 여러 Seed가 반복 FP
- lexical shortcut이 명확
- label boundary는 명확하지만 negative coverage가 부족
- 새로운 독립 사례를 현실적으로 만들 수 있음

### Label guide / GT 재검토 우선

다음이 많으면 고려:

- label ambiguity
- suspected annotation error
- reviewer disagreement

### Weighted BCE 검토

다음 경우에만 우선순위를 높임:

- 오류 분석상 데이터 경계보다 optimization imbalance가 더 설득력 있음
- 단순 FP pattern 부족만으로 설명되지 않음

Weighted BCE는 효과 없다고 미리 단정하지 않는다.
다만 원인 분석 전에 loss를 변경하지 않는다.

### Alternative split

특정 split 의존성이 여전히 주요 위험이라고 판단될 경우
최종 설정 1회 sensitivity check로 고려한다.

이는 full split variance estimation이 아니라 방향성 확인이다.

---

## 최종 산출물

사람 검토 후 최소한 아래를 보고한다.

- label별 FP/FN 개수
- error category 비율
- 여러 Seed에서 반복된 error pattern
- annotation ambiguity 비율
- suspected annotation error 수
- hard-negative 후보 유형
- V6 유지 / V6.1 결정과 근거
